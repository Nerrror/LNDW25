def caption_integrated_gradients(pil_image, selected_idx, pixel_values, caption_model, caption_processor, caption_device, caption_tokenizer,
                              steps=64, chunk_size=8, progress=None):
    """Compute integrated gradients for caption model for a specific token.
    
    Args:
        selected_idx: Index in the filtered token list (non-special tokens only)
    """
    import torch
    
    caption_model.eval()
    
    # Use pre-computed pixel values (already on the correct device)
    x_in = pixel_values
    
    # Generate tokens to match UI selection (run with same parameters)
    with torch.no_grad():
        # Run generation with the same parameters to get full sequence
        output_ids = caption_model.generate(
            pixel_values=x_in,
            max_length=16,
            num_beams=4,
        )
        
        # Filter special tokens to map UI index to actual token position
        token_positions = []
        for i, token_id in enumerate(output_ids[0]):
            if token_id.item() not in caption_tokenizer.all_special_ids:
                token_positions.append(i)
        
        if selected_idx >= len(token_positions):
            selected_idx = len(token_positions) - 1
        
        # Get the actual position in the sequence (including special tokens)
        actual_position = token_positions[selected_idx]
        token_id = output_ids[0][actual_position].item()
        token_text = caption_tokenizer.decode([token_id])
    
    # Baseline: black image in normalized space
    try:
        mean = torch.tensor(caption_processor.image_mean, device=caption_device, dtype=x_in.dtype).view(1, -1, 1, 1)
        std = torch.tensor(caption_processor.image_std, device=caption_device, dtype=x_in.dtype).view(1, -1, 1, 1)
        x_base = (-mean / std).expand_as(x_in)
    except Exception:
        x_base = torch.zeros_like(x_in)  # Fallback
    
    # Compute grads along the path in small chunks to limit memory
    alphas = torch.linspace(0, 1, steps, device=caption_device, dtype=x_in.dtype)
    grad_sum = torch.zeros_like(x_in)
    
    # Target is the log probability of the selected token
    target_token_id = token_id
    
    for i in range(0, steps, chunk_size):
        a_chunk = alphas[i:i+chunk_size].view(-1, 1, 1, 1)            # (b,1,1,1)
        path_chunk = x_base + a_chunk * (x_in - x_base)               # (b,C,H,W)
        path_chunk.requires_grad_(True)
        
        # Run encoder-decoder but only care about actual_position prediction
        encoder_outputs = caption_model.encoder(path_chunk, return_dict=True)
        
        # Use all previous tokens as context
        decoder_input_ids = output_ids[0][:actual_position].unsqueeze(0).expand(path_chunk.shape[0], -1).to(caption_device)
        
        outputs = caption_model(
            pixel_values=None,  # Already passed to encoder
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            return_dict=True
        )
        
        # Get logits for the token we care about
        # Note: actual_position - 1 because we're predicting the next token after the context
        token_logits = outputs.logits[:, actual_position-1, target_token_id]  # pred of target token
        
        grads_chunk = torch.autograd.grad(
            outputs=token_logits,
            inputs=path_chunk,
            grad_outputs=torch.ones_like(token_logits),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        
        # Sum over batch dimension; we'll divide by total steps later
        grad_sum += grads_chunk.sum(dim=0, keepdim=True)
        
        # Optional: update progress bar to keep Streamlit connection alive
        if progress is not None:
            progress.progress(min(1.0, (i + grads_chunk.shape[0]) / steps))
    
    avg_grads = grad_sum / steps                                       # (1,C,H,W)
    ig_attr = (x_in - x_base) * avg_grads                              # (1,C,H,W)
    
    heat = ig_attr.squeeze(0).sum(dim=0).detach().cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    
    return heat, token_text, actual_position