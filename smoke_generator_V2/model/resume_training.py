import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
import os
from unet import UNET
from custom_dataset import CustomDataset
from sklearn.metrics import f1_score, jaccard_score
import pickle
import argparse

##-------------------------------------------------------------------------------------------------------------------
##                         Resuming training from existing checkpoint 
##-------------------------------------------------------------------------------------------------------------------

def main(run_id,checkpoint_path,last_epoch):
  wandb.login(key='124b22523d2147cafbedea2a2648c9c6a1450fcd')
  api = wandb.Api()
  run = api.run(run_id)

  run_history = run.history()
  metric_name1 = "Validation Loss"  # Replace with the actual metric name
  metric_name2 = "Dice Coefficient (F1 Score)"
  metric_name3 = "IoU"
  # Find the best (minimum) value for the specified metric
  best_val_loss = min(run_history[metric_name1])
  best_val_dice = max(run_history[metric_name2])
  best_val_iou = max(run_history[metric_name3])
  print(f"Best {metric_name1}: {best_val_loss}")
  print(f"Best {metric_name2}: {best_val_dice}")
  print(f"Best {metric_name3}: {best_val_iou}")
  if last_epoch == "None":
    last_epoch = run_history.index[-1]
  config = run.config
  print(f"Config: {config}")
  # Print the last epoch
  print(f"Last Epoch: {last_epoch}")

  # Can change when finetuning
  log_interval = 1
  save_interval = 25
   # Define the root directory where your image and mask folders are located
  root_dir = r'/content/drive/MyDrive/smoke_generator_V2/smoke_dataset_V1'

  seed = config["seed"]
  batch_size = config["batch_size"]
  learning_rate = config["learning_rate"]
  criterion_name = config["criterion"]
  optimizer_name = config["optimizer"]
  num_epochs = config["num_epochs"]
  weight_decay = config["weight_decay"]
  model_name = config["model_name"]
  img_size = config["image_size"]
  dataset_name = root_dir.split("/")[-1]
  dataset_length = config["dataset_length"]

  # Check if GPU is available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("-----------------------------------------------------------------------")
  print("Resume Training Starting on device:" "gpu" if torch.cuda.is_available() else "cpu")
  print("-----------------------------------------------------------------------")


  wandb.init(project='albatros_smoke_segmentation',id=run_id.split("/")[-1],resume="must",config=config)
  run_name = wandb.run.name
  print("-----------------------------------------------------------------------")
  print(f"Resume Training run: {run_name}")
  print("-----------------------------------------------------------------------")


  # Set a random seed for reproducibility
  torch.manual_seed(seed)
  np.random.seed(seed)

  #--Weight directory
  file_dir = os.path.dirname(os.path.abspath(__file__))
  weight_dir = os.path.join(file_dir, "../weight",run_name)
  if not os.path.exists(weight_dir):
      # If it doesn't exist, create it
      os.makedirs(weight_dir,exist_ok=True)
      print("-----------------------------------------------------------------------")
      print(f"{weight_dir} has been created")
      print("-----------------------------------------------------------------------")
  else :
    print("-----------------------------------------------------------------------")
    print(f"Model weights can be found at: {weight_dir}")
    print("-----------------------------------------------------------------------")

  # Define data transforms if needed
  train_transform = transforms.Compose([
      transforms.Resize(img_size),
      transforms.ToTensor(),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
  ])

  val_test_transform = transforms.Compose([
      transforms.Resize(img_size),
      transforms.ToTensor(),
  ])

  dataset = CustomDataset(root_dir, transform=train_transform)


  indices = list(range(len(dataset)))
  indices = indices[:dataset_length]
  dataset = CustomDataset(root_dir, transform=train_transform,indices=indices)


  train_size = int(0.8 * len(dataset))
  val_size = int(0.1 * len(dataset))
  test_size = len(dataset) - train_size - val_size

  train_indices = indices[:train_size]
  val_indices = indices[train_size:train_size + val_size]
  test_indices = indices[train_size + val_size:]

  # Apply different transforms based on the split
  train_dataset = CustomDataset(root_dir, transform=train_transform, indices=train_indices)
  val_dataset = CustomDataset(root_dir, transform=val_test_transform, indices=val_indices)
  test_dataset = CustomDataset(root_dir, transform=val_test_transform, indices=test_indices)


  print("-----------------------------------------------------------------------")
  print(f"Dataset size:{len(dataset)}")
  print(f"Train size:{train_size}")
  print(f"Val size:{val_size}")
  print(f"Test size:{test_size}")
  print("-----------------------------------------------------------------------")


  # Create DataLoaders for the datasets
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True,pin_memory=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size,pin_memory=True)  

  # Initialize the UNet model and move it to the GPU if available
  in_channels = 3  # Number of input channels (e.g., RGB image)
  out_channels = 1  # Number of output channels (e.g., binary segmentation)
  if model_name == "Unet":
    model = UNET(in_channels, out_channels).to(device)
    model.load_state_dict(torch.load(checkpoint_path,map_location = torch.device(device)))

  # Define loss function and optimizer
  if criterion_name == 'BCEWithLogitsLoss':
      criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
  if optimizer_name == 'Adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

  # Training loop
  best_model_weights_path = None

  for epoch in range(last_epoch,num_epochs):
      model.train()  # Set the model to training mode
      train_loss = 0.0
      train_dice_scores = []
      train_iou_scores = []

      for batch_idx, batch in enumerate(train_loader):
          images = batch['image'].to(device)
          masks = batch['mask'].to(device)

          optimizer.zero_grad()  # Zero the gradients

          # Forward pass
          outputs = model(images)
    
          # Calculate the loss
          loss = criterion(outputs, masks)

          # Backpropagation and optimization
          loss.backward()
          optimizer.step()

          train_loss += loss.item()

          train_outputs = (torch.sigmoid(outputs) > 0.5).float()
          train_masks = (masks > 0.5).float()

          # Calculate Dice Coefficient (F1 Score) on the GPU
          dice = f1_score(train_masks.view(-1).cpu().numpy(), (torch.sigmoid(train_outputs) > 0.5).view(-1).cpu().numpy())
          train_dice_scores.append(dice)

          # Calculate Intersection over Union (IoU) on the GPU
          iou = jaccard_score(train_masks.view(-1).cpu().numpy(), (torch.sigmoid(train_outputs) > 0.5).view(-1).cpu().numpy())
          train_iou_scores.append(iou)

      avg_train_loss = train_loss / len(train_loader)
      avg_train_dice = sum(train_dice_scores) / len(train_dice_scores)
      avg_train_iou = sum(train_iou_scores) / len(train_iou_scores)

      wandb.log({"Train Loss": avg_train_loss, "Train Dice Coefficient (F1 Score)": avg_train_dice, "Train IoU": avg_train_iou}, step=epoch)
      print(f'Training Loop of Epoch:{epoch+1}/{num_epochs} over average training loss : {avg_train_loss}')

      # Validation loop
      model.eval()  # Set the model to evaluation mode
      val_loss = 0.0
      dice_scores = []
      iou_scores = []

      with torch.no_grad():
          for val_batch in val_loader:
            val_images = val_batch['image'].to(device)
            val_masks = val_batch['mask'].to(device)

            # Forward pass
            val_outputs = model(val_images)

            # Apply sigmoid activation and threshold to obtain binary predictions on the GPU
            val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
            val_masks = (val_masks > 0.5).float()

            # Calculate the validation loss
            val_loss += criterion(val_outputs, val_masks).item()

            # Calculate Dice Coefficient (F1 Score) on the GPU
            dice = f1_score(val_masks.view(-1).cpu().numpy(), val_outputs.view(-1).cpu().numpy())
            dice_scores.append(dice)

            # Calculate Intersection over Union (IoU) on the GPU
            iou = jaccard_score(val_masks.view(-1).cpu().numpy(), val_outputs.view(-1).cpu().numpy())
            iou_scores.append(iou)

      avg_val_loss = val_loss / len(val_loader)
      avg_dice_score = sum(dice_scores) / len(dice_scores)
      avg_iou_score = sum(iou_scores) / len(iou_scores)

      # Log metrics to wandb
      wandb.log({"Validation Loss": avg_val_loss, "Dice Coefficient (F1 Score)": avg_dice_score, "IoU": avg_iou_score},step=epoch)

      # Log images for one sample
      sample_images = val_images[:1]  # Take the first image from the validation set
      sample_masks = val_masks[:1]
      sample_outputs = val_outputs[:1]

      # Convert tensor images to numpy arrays
      sample_images = sample_images.permute(0, 2, 3, 1).cpu().numpy()
      sample_masks = sample_masks.permute(0, 2, 3, 1).cpu().numpy()
      sample_outputs = (sample_outputs > 0.5).permute(0, 2, 3, 1).cpu().numpy()

      # Plot the three images together
      fig, axes = plt.subplots(1, 3, figsize=(15, 5))
      axes[0].imshow(sample_images[0])
      axes[0].set_title('Val Input Image')
      axes[1].imshow(sample_masks[0], cmap='gray')
      axes[1].set_title('Val Mask Input')
      axes[2].imshow(sample_outputs[0], cmap='gray')
      axes[2].set_title('Val Mask Output')
      plt.close()
      # Log the combined plot to WandB
      wandb.log({"Sample Images": [wandb.Image(fig)]}, step=epoch)

      # Print to terminal
      print(f" Validation Loss Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Dice: {avg_dice_score:.4f}, IoU: {avg_iou_score:.4f}")

      # Save model weights
      if (epoch + 1) % save_interval == 0:
          model_weights_path = os.path.join(weight_dir, f"model_epoch_{epoch + 1}.pt")
          torch.save(model.state_dict(), model_weights_path)
      # Save one model weights per metrics
      if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          best_model_weights_path = os.path.join(weight_dir, "best_model_val_loss.pt")
          torch.save(model.state_dict(), best_model_weights_path)
      if avg_dice_score > best_val_dice:
          best_val_dice = avg_dice_score
          best_model_weights_path = os.path.join(weight_dir, "best_model_val_dice.pt")
          torch.save(model.state_dict(), best_model_weights_path)
      if avg_iou_score > best_val_iou:
          best_val_iou = avg_iou_score        
          best_model_weights_path = os.path.join(weight_dir, "best_model_val_iou.pt")
          torch.save(model.state_dict(), best_model_weights_path)

  print("-----------------------------------------------------------------------")
  print("Training complete!")
  print("-----------------------------------------------------------------------")

  # Testing loop
  model.eval()  # Set the model to evaluation mode
  test_loss = 0.0
  test_dice_scores = []
  test_iou_scores = []

  with torch.no_grad():
      for test_batch in test_loader:
          test_images = test_batch['image'].to(device)
          test_masks = test_batch['mask'].to(device)

          # Forward pass
          test_outputs = model(test_images)

          # Apply sigmoid activation and threshold to obtain binary predictions on the GPU
          test_outputs = (torch.sigmoid(test_outputs) > 0.5).float()
          test_masks = (test_masks > 0.5).float()

          # Calculate the test loss
          test_loss += criterion(test_outputs, test_masks).item()

          # Calculate Dice Coefficient (F1 Score) on the GPU
          dice = f1_score(test_masks.view(-1).cpu().numpy(), test_outputs.view(-1).cpu().numpy())
          test_dice_scores.append(dice)

          # Calculate Intersection over Union (IoU) on the GPU
          iou = jaccard_score(test_masks.view(-1).cpu().numpy(), test_outputs.view(-1).cpu().numpy())
          test_iou_scores.append(iou)

  avg_test_loss = test_loss / len(test_loader)
  avg_test_dice = sum(test_dice_scores) / len(test_dice_scores)
  avg_test_iou = sum(test_iou_scores) / len(test_iou_scores)

  # Log metrics to WandB for the test set
  wandb.log({"Test Loss": avg_test_loss, "Test Dice Coefficient (F1 Score)": avg_test_dice, "Test IoU": avg_test_iou})

  print("-----------------------------------------------------------------------")
  print("Testing Loop")
  print(f"Test Loss: {avg_test_loss:.4f}")
  print(f"Test Dice: {avg_test_dice:.4f}")
  print(f"Test IoU: {avg_test_iou:.4f}")
  print("-----------------------------------------------------------------------")

if __name__ == "__main__":  
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_id', '-rID', type=str, required = True, help='run name path')
  parser.add_argument('--checkpoint_path', '-ckpth', type=str, required= True , default='', help='Path to the checkpoint file with the models weights')
  parser.add_argument('--last_epoch', '-le', type=int, required = False,default="None", help='run name path')
  args = parser.parse_args()
  
  main(args.run_id,args.checkpoint_path,args.last_epoch)
