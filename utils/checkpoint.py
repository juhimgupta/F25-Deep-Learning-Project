import torch
import os

def load_checkpoint(
    unet,
    scheduler=None,# this is the *training* LR scheduler if we need to use it; inference can pass None
    vae=None,
    class_embedder=None,
    optimizer=None,
    checkpoint_path="checkpoints/checkpoint.pth",
    map_location="cpu",
):
    print(f"[load_checkpoint] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_location)

    # UNET
    print("[load_checkpoint] Loading UNet weights")
    if "unet_state_dict" in checkpoint:
        unet_state = checkpoint["unet_state_dict"]
    elif "unet" in checkpoint:
        unet_state = checkpoint["unet"]
    else:
        raise KeyError(
            f"Neither 'unet_state_dict' nor 'unet' found in checkpoint keys: {checkpoint.keys()}"
        )
    unet.load_state_dict(unet_state)

    # OPTIMIZER (optional, for training resume) 
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            print("[load_checkpoint] Loading optimizer state")
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(f"[load_checkpoint] Failed to load optimizer state: {e}")

    # LR SCHEDULER
    if scheduler is not None and "lr_scheduler" in checkpoint:
        try:
            print("[load_checkpoint] Loading lr_scheduler state")
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
        except Exception as e:
            print(f"[load_checkpoint] Failed to load lr_scheduler state: {e}")

    # VAE 
    if vae is not None:
        for k in ["vae", "vae_state_dict"]:
            if k in checkpoint and checkpoint[k] is not None:
                try:
                    print(f"[load_checkpoint] Loading VAE state from '{k}'")
                    vae.load_state_dict(checkpoint[k])
                except Exception as e:
                    print(f"[load_checkpoint] Failed to load VAE state from '{k}': {e}")
                break
        else:
            print("[load_checkpoint] No VAE weights found in checkpoint")

    # CLASS EMBEDDER
    if class_embedder is not None:
        for k in ["class_embedder", "class_embedder_state_dict"]:
            if k in checkpoint and checkpoint[k] is not None:
                try:
                    print(f"[load_checkpoint] Loading class_embedder from '{k}'")
                    class_embedder.load_state_dict(checkpoint[k])
                except Exception as e:
                    print(f"[load_checkpoint] Failed to load class_embedder from '{k}': {e}")
                break
        else:
            print("[load_checkpoint] No class_embedder weights found in checkpoint")

    print("[load_checkpoint] Done.")
    return checkpoint
    
    
        

def save_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, epoch=None, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define checkpoint file name
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')

    checkpoint = {
        'unet_state_dict': unet.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    if vae is not None:
        checkpoint['vae_state_dict'] = vae.state_dict()
    
    if class_embedder is not None:
        checkpoint['class_embedder_state_dict'] = class_embedder.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    # Manage checkpoint history
    manage_checkpoints(save_dir, keep_last_n=10)


def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")