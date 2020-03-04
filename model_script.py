from pipeline.train import train_fastfcn_mod

# -----------------------------------------
# -------------- Main ---------------------
# -----------------------------------------

# Use this to quickly test functions.

if __name__=='__main__':
    
    train_fastfcn_mod(
        num_epochs=5, reporting_int=5,
        batch_size=16, MODEL_NICKNAME='five_epoch_single_region'
        )

# # Predict
# with torch.no_grad():
#     output = model(img_tensor.view(-1, 3, 1024, 1024))[2]
# np_pred = torch.max(output, 1)[1].cpu().numpy() * 255
# out_img = Image.fromarray(np_pred.squeeze().astype('uint8'))