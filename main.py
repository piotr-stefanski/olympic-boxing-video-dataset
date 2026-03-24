import matplotlib.pyplot as plt
import cv2
import os
import torch
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
from engine import train_one_epoch, evaluate
from dataloader import CocoDetectionDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f'Used {device=}')

# Transform PIL image --> PyTorch tensor
def get_transform():
    return ToTensor()

def save_few_examples(data_loader, output_dir, num_examples=5):
    """
    Save a few sample images with bounding boxes to the output directory.
    
    Args:
        data_loader: DataLoader to get samples from
        output_dir: Directory to save the output images
        num_examples: Number of examples to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get one batch
    images, targets = next(iter(data_loader))
    
    saved_count = 0
    # Loop through the batch and draw bounding boxes and labels
    for i in range(min(len(images), num_examples)):
        # CxHxW --> HxWxC
        image = images[i].permute(1, 2, 0).numpy()
        # Rescale
        image = (image * 255).astype(np.uint8)
        # Convert RGB to BGR for cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get bounding box coordinates and labels
        boxes = targets[i]['boxes']
        labels = targets[i]['labels']
    
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Class {label.item()}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Convert BGR back to RGB for saving
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save image to file instead of displaying
        output_path = os.path.join(output_dir, f"sample_{i + 1:03d}.png")
        plt.figure(figsize=(16, 12))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(f"Sample {i + 1}")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Saved: {output_path}")
        saved_count += 1
    
    print(f"Saved {saved_count} example images to {output_dir}")


def main():
    # Load training datasets (folds 1, 2, and 3)
    image_dir = '../datasets/olympic-boxing-video-dataset/coco_images'
    annotations_dir = '../datasets/olympic-boxing-video-dataset/annotations'
    
    fold_1_dataset = CocoDetectionDataset(
        image_dir=image_dir,
        annotation_path=f'{annotations_dir}/annotations_fold_1.json',
        transforms=get_transform()
    )
    fold_2_dataset = CocoDetectionDataset(
        image_dir=image_dir,
        annotation_path=f'{annotations_dir}/annotations_fold_2.json',
        transforms=get_transform()
    )
    fold_3_dataset = CocoDetectionDataset(
        image_dir=image_dir,
        annotation_path=f'{annotations_dir}/annotations_fold_3.json',
        transforms=get_transform()
    )
    fold_4_dataset = CocoDetectionDataset(
        image_dir=image_dir,
        annotation_path=f'{annotations_dir}/annotations_fold_4.json',
        transforms=get_transform()
    )
    train_dataset = ConcatDataset([fold_1_dataset, fold_2_dataset, fold_3_dataset, fold_4_dataset])
    
    # Load validation dataset
    val_dataset = CocoDetectionDataset(
        image_dir=image_dir,
        annotation_path=f'{annotations_dir}/annotations_fold_5.json',
        transforms=get_transform()
    )
    
    # Load dataset with DataLoaders, you can change batch_size 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True, prefetch_factor=4,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            num_workers=4, pin_memory=True, prefetch_factor=4,
                            collate_fn=lambda x: tuple(zip(*x)))

    # Load a pre-trained Faster R-CNN model with ResNet50 backbone and FPN, , you change this 
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Number of classes in the dataset (including background)
    # +1 for bg class
    num_classes = len(fold_1_dataset.coco.getCatIds()) + 1 
    print(f'Number of classes: {num_classes}')
    
    # Number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    """  
    Number of classes must be equal to your label number
    """
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Move the model to the GPU for faster training
    model.to(device)

    # Get parameters that require gradients (the model's trainable parameters)
    params = [p for p in model.parameters() if p.requires_grad]
 
    # Define the optimizer SGD(Stochastic Gradient Descent) 
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Number of epochs for training
    num_epochs = 100

    writer = SummaryWriter(log_dir="output/tensorboard_logs")

    # Loop through each epoch
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
        # Train the model for one epoch, printing status every 25 iterations
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)

        # Log training loss
        avg_loss = metric_logger.meters['loss'].global_avg
        writer.add_scalar('Loss/train', avg_loss, epoch)
    
        # Evaluate the model only on the validation dataset, not training
        coco_evaluator = evaluate(model, val_loader, device=device)

        # Log COCO evaluation metrics
        coco_stats = coco_evaluator.coco_eval['bbox'].stats
        writer.add_scalar('AP/IoU_0.50_0.95_all_maxDets_100', coco_stats[0], epoch)
        writer.add_scalar('AP/IoU_0.50_all_maxDets_100', coco_stats[1], epoch)
        writer.add_scalar('AR/IoU_0.50_0.95_all_maxDets_10', coco_stats[6], epoch)
    
        if (epoch + 1) % 10 == 0:
            # save the model after each epoch
            print(f"Saving model after epoch {epoch + 1}")
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

    writer.close()

if __name__ == "__main__":
    main()
