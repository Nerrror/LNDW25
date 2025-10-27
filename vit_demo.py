import streamlit as st
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import ViTModel, ViTImageProcessor
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from transformers import ViTForImageClassification
from typing import Optional, Any, List, Tuple

device = "cpu"

# ------------------------
# Load pretrained ViT model
# ------------------------
model_name = "google/vit-base-patch16-224"
model = ViTModel.from_pretrained(
    model_name,
    output_attentions=True,
    add_pooling_layer=False  # disables the pooler entirely
)
processor = ViTImageProcessor.from_pretrained(model_name)

st.title("👁 Vision Transformer Demo: Patches, Aufmerksamkeit & Rollout")

# ------------------------
# Load pretrained ViT classifier (for gradient-based attributions)
# ------------------------
classifier_model_name = model_name  # same backbone family
classifier_model = ViTForImageClassification.from_pretrained(classifier_model_name)
classifier_model.eval()
    # Freeze classifier weights to avoid computing/storing parameter gradients during IG
for _p in classifier_model.parameters():
    _p.requires_grad_(False)

# ------------------------
# Load pretrained captioning model
# ------------------------
caption_model_name = "nlpconnect/vit-gpt2-image-captioning"
caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_name)
caption_processor = ViTImageProcessor.from_pretrained(caption_model_name)
caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_name)

# Ensure caption model is fully on a real device (not 'meta')
caption_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(caption_device)
caption_model.eval()

# Freeze caption model weights for IG
for _p in caption_model.parameters():
    _p.requires_grad_(False)

def generate_caption(img: Image.Image):
    """Generate a text description for the given image.
    Returns:
        tuple: (caption_text, token_texts, pixel_values, token_ids)
    """
    cap_inputs = caption_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        cap_pixel_values = cap_inputs["pixel_values"].to(caption_device)
        output_ids = caption_model.generate(
            pixel_values=cap_pixel_values, 
            max_length=16, 
            num_beams=4
        )
    caption_text = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Get individual tokens (for IG token selection)
    # Filter out special tokens like [CLS], [SEP], etc.
    token_ids = [t.item() for t in output_ids[0] if t.item() not in caption_tokenizer.all_special_ids]
    token_texts = [caption_tokenizer.decode([t]) for t in token_ids]
    
    return caption_text, token_texts, cap_pixel_values, token_ids

def attention_rollout(attn_list):
    """
    Perform attention rollout across layers.
    attentions: list of (batch, heads, tokens, tokens)
    returns: (tokens, tokens) aggregated attention
    """
    num_tokens = attn_list[0].size(-1)
    result = torch.eye(num_tokens)
    
    for attn_mat in attn_list:
        # Average heads
        attn_heads = attn_mat[0].mean(dim=0)
        # Add identity to preserve residual connection
        attn_heads = attn_heads + torch.eye(num_tokens)
        # Normalize
        attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
        # Multiply with previous result
        result = torch.matmul(attn_heads, result)
    
    return result.numpy()

def integrated_gradients(pil_image: Image.Image, steps: int = 64, target_class: Optional[int] = None, chunk_size: int = 8, progress: Optional[Any] = None):
    classifier_model.eval()

    ig_inputs = processor(images=pil_image, return_tensors="pt")
    x_in = ig_inputs["pixel_values"].to(device)

    with torch.no_grad():
        logits = classifier_model(x_in).logits
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=-1).item())

    id2label = getattr(classifier_model.config, "id2label", None)
    if isinstance(id2label, dict):
        predicted_label = id2label.get(target_class, str(target_class))
    elif isinstance(id2label, (list, tuple)) and target_class < len(id2label):
        predicted_label = id2label[target_class]
    else:
        predicted_label = str(target_class)

    # Baseline: schwarzes Bild im normalisierten Raum
    try:
        mean = torch.tensor(processor.image_mean, device=device, dtype=x_in.dtype).view(1, -1, 1, 1)
        std  = torch.tensor(processor.image_std,  device=device, dtype=x_in.dtype).view(1, -1, 1, 1)
        x_base = (-mean / std).expand_as(x_in)
    except Exception:
        x_base = torch.zeros_like(x_in)  # Fallback

    # Compute grads along the path in small chunks to limit memory
    alphas = torch.linspace(0, 1, steps, device=device, dtype=x_in.dtype)
    grad_sum = torch.zeros_like(x_in)
    for i in range(0, steps, chunk_size):
        a_chunk = alphas[i:i+chunk_size].view(-1, 1, 1, 1)            # (b,1,1,1)
        path_chunk = x_base + a_chunk * (x_in - x_base)               # (b,C,H,W)
        path_chunk.requires_grad_(True)

        logits_chunk = classifier_model(path_chunk).logits             # (b,num_cls)
        scores_chunk = logits_chunk[:, target_class]                   # (b,)
        grads_chunk = torch.autograd.grad(
            outputs=scores_chunk,
            inputs=path_chunk,
            grad_outputs=torch.ones_like(scores_chunk),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]                                                           # (b,C,H,W)

        # Sum over batch dimension; we'll divide by total steps later
        grad_sum += grads_chunk.sum(dim=0, keepdim=True)

        # Optional: update progress bar to keep Streamlit connection alive
        if progress is not None:
            progress.progress(min(1.0, (i + grads_chunk.shape[0]) / steps))

    avg_grads = grad_sum / steps                                       # (1,C,H,W)
    ig_attr = (x_in - x_base) * avg_grads                              # (1,C,H,W)

    heat = ig_attr.squeeze(0).sum(dim=0).detach().cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat, predicted_label, target_class


# ------------------------
# Upload image
# ------------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img_orig = Image.open(uploaded_file).convert("RGB")
    st.image(img_orig, caption="Originalbild", use_container_width=True)
    
    # Show patches
    patch_size = 16
    transform = T.Compose([
        T.Resize(224),        # resize shorter side to 224, keep aspect ratio
        T.CenterCrop(224)     # crop longer side to 224
    ])


    img_resized = transform(img_orig)
    img_array = np.array(img_resized)

    # --- Generate description ---
    generated_caption, token_texts, cap_pixel_values, token_ids = generate_caption(img_resized)
    st.subheader("Bild-zu-Text Modell:")
    st.markdown(f"<h3 style='text-align: center;'>{generated_caption}</h3>", unsafe_allow_html=True)

    fig, ax = plt.subplots()
    ax.imshow(img_array)
    h, w, _ = img_array.shape
    for y in range(1, h, patch_size):
        ax.axhline(y - 0.25, color="white", linewidth=0.5)
    for x in range(1, w, patch_size):
        ax.axvline(x - 0.25, color="white", linewidth=0.5)
    ax.set_title("Bild aufgeteilt in Patches")
    ax.axis("off")
    st.pyplot(fig)

    # Forward through model
    inputs = processor(images=img_resized, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions  # list of (batch, heads, tokens, tokens)

    # --- Classification result ---
    with torch.no_grad():
        cls_outputs = classifier_model(**inputs)
        cls_logits = cls_outputs.logits
        cls_probs = torch.softmax(cls_logits, dim=-1)
        cls_idx = int(torch.argmax(cls_probs, dim=-1).item())
        cls_conf = float(cls_probs[0, cls_idx].item())
        
        id2label = getattr(classifier_model.config, "id2label", None)
        if isinstance(id2label, dict):
            cls_label = id2label.get(cls_idx, str(cls_idx))
        elif isinstance(id2label, (list, tuple)) and cls_idx < len(id2label):
            cls_label = id2label[cls_idx]
        else:
            cls_label = str(cls_idx)
    
    st.subheader("Klassifikationsergebnis:")
    st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{cls_label}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Konfidenz: {cls_conf:.2%}</h3>", unsafe_allow_html=True)

        # --- Add slider for layer selection ---
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    layer_idx = st.slider("Transformer-Schicht auswählen", 1, num_layers, num_layers)

    # --- Add toggles for modes (mutually exclusive) ---
    # Initialize session state keys
    for key in ("avg_mode", "rollout_mode", "ig_mode", "caption_ig_mode"):
        if key not in st.session_state:
            st.session_state[key] = False

    def _select_mode(selected_key: str):
        # Ensure only one mode is active at a time
        for k in ("avg_mode", "rollout_mode", "ig_mode", "caption_ig_mode"):
            st.session_state[k] = (k == selected_key) and st.session_state[k]

    st.subheader("🔍 Visualisierungsmodus")
    col1, col2 = st.columns(2)
    with col1:
        avg_mode = st.checkbox(
            "Gemittelte Aufmerksamkeit über alle Köpfe anzeigen",
            key="avg_mode",
            on_change=_select_mode,
            args=("avg_mode",),
        )
        rollout_mode = st.checkbox(
            "Attention Rollout verwenden (aggregiert über Schichten)",
            key="rollout_mode",
            on_change=_select_mode,
            args=("rollout_mode",),
        )
    with col2:
        ig_mode = st.checkbox(
            "Integrierte Gradienten verwenden (Klassifikation)",
            key="ig_mode",
            on_change=_select_mode,
            args=("ig_mode",),
        )
        caption_ig_mode = st.checkbox(
            "Integrierte Gradienten verwenden (Bildbeschreibung)",
            key="caption_ig_mode",
            on_change=_select_mode,
            args=("caption_ig_mode",),
        )

    # Settings based on selected mode
    if st.session_state.ig_mode or st.session_state.caption_ig_mode:
        col1, col2 = st.columns(2)
        with col1:
            ig_steps = st.slider("Schritte für Integrierte Gradienten", min_value=16, max_value=256, value=64, step=16)
        with col2:
            # Compute valid chunk sizes (limit options to keep UI clean)
            valid_chunks = [2, 4, 8, 16, 32]
            valid_chunks = [c for c in valid_chunks if c <= ig_steps]
            ig_chunk = st.selectbox("IG Chunk-Größe", options=valid_chunks, index=min(2, len(valid_chunks)-1))
    
    # Token selection for Caption IG
    if st.session_state.caption_ig_mode:
        # Import the caption IG function
        from caption_integrated_gradients import caption_integrated_gradients
        
        # Extract non-special tokens
        token_options = [f"{i}: {token}" for i, token in enumerate(token_texts)]
        selected_token_idx = st.selectbox("Token für Attribution auswählen", 
                                         options=range(len(token_options)),
                                         format_func=lambda i: token_options[i])
    
    if st.session_state.caption_ig_mode:
        # Compute Caption IG
        with st.spinner("Berechne Integrierte Gradienten für Bildbeschreibung..."):
            prog = st.progress(0.0)
            ig_map, token_text, actual_position = caption_integrated_gradients(
                img_resized, selected_token_idx, cap_pixel_values, 
                caption_model, caption_processor, caption_device, caption_tokenizer,
                steps=ig_steps, chunk_size=ig_chunk, progress=prog
            )
            
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_resized)
        im = ax.imshow(ig_map, alpha=0.6, cmap="jet")
        ax.set_title(f"Bildbeschreibung IG: Beitrag zum Token '{token_text}'")
        ax.axis("off")
        plt.colorbar(im)
        st.pyplot(fig)
            
    elif st.session_state.ig_mode:
        # Compute Classification IG
        with st.spinner("Berechne Integrierte Gradienten für Klassifikation..."):
            prog = st.progress(0.0)
            ig_map, pred_label, pred_class = integrated_gradients(
                img_resized, steps=ig_steps, chunk_size=ig_chunk, progress=prog
            )
            
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_resized)
        im = ax.imshow(ig_map, alpha=0.6, cmap="jet")
        ax.set_title(f"Klassifikation IG: Beitrag zur Klasse '{pred_label}'")
        ax.axis("off")
        plt.colorbar(im)
        st.pyplot(fig)

    elif rollout_mode:
        # Rollout across all layers
        attn_roll = attention_rollout(attentions)
        patch_attn = attn_roll[0, 1:].reshape(14, 14)

        fig, ax = plt.subplots()
        ax.imshow(img_resized)
        ax.imshow(patch_attn, cmap="jet", alpha=0.5, extent=(0,224,224,0))
        ax.set_title("Attention Rollout (Alle Schichten)")
        ax.axis("off")
        st.pyplot(fig)

    elif avg_mode:
        # Average across all heads in the chosen layer
        attn_avg = attentions[layer_idx - 1][0].mean(dim=0).numpy()
        patch_attn = attn_avg[0, 1:].reshape(14, 14)

        fig, ax = plt.subplots()
        ax.imshow(img_resized)
        ax.imshow(patch_attn, cmap="jet", alpha=0.5, extent=(0,224,224,0))
        ax.set_title(f"Gemittelte Aufmerksamkeitskarte (Schicht {layer_idx})")
        ax.axis("off")
        st.pyplot(fig)

    else:
        # Show all heads in the chosen layer
        layer_attn = attentions[layer_idx - 1][0].numpy()  # shape: (heads, tokens, tokens)

        cols = 4  # grid layout (e.g. 3 rows × 4 cols for 12 heads)
        rows = int(np.ceil(num_heads / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten()

        for head_idx in range(num_heads):
            head_attn = layer_attn[head_idx]
            patch_attn = head_attn[0, 1:].reshape(14, 14)

            axes[head_idx].imshow(img_resized)
            axes[head_idx].imshow(patch_attn, cmap="jet", alpha=0.5, extent=(0,224,224,0))
            axes[head_idx].set_title(f"Schicht {layer_idx}, Kopf {head_idx+1}")
            axes[head_idx].axis("off")

        # Hide unused subplots if num_heads not multiple of cols
        for j in range(num_heads, len(axes)):
            axes[j].axis("off")

        st.pyplot(fig)

