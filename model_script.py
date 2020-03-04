from pipeline.train import train_fastfcn_mod

# -----------------------------------------
# -------------- Main ---------------------
# -----------------------------------------

# Use this to quickly test functions.

if __name__=='__main__':
    
    train_fastfcn_mod(
        num_epochs=5, reporting_int=5,
        batch_size=16, MODEL_NICKNAME=five_epoch_single_region
        )