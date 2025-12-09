"""
Computer Vision Model for Robotics Control Decision Making.

This module provides a lightweight CNN model for making control decisions
based on camera frames. Uses a pretrained MobileNetV2 as the backbone.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import numpy as np
import cv2


class SimpleControlModel(nn.Module):
    """
    Simple CNN model for robotics control decisions.
    
    Uses a pretrained MobileNetV2 backbone and fine-tunes it for
    binary classification: object_detected vs no_object.
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of output classes (default: 2 for binary classification)
            pretrained: Whether to use pretrained weights (default: True)
        """
        super(SimpleControlModel, self).__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = mobilenet_v2(weights=None)
        
        # Replace the classifier head for binary classification
        # MobileNetV2's last layer expects 1280 features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.backbone(x)


def load_model(model_path=None, device='cpu', pretrained=True):
    """
    Load the model with optional pretrained weights or from a checkpoint.
    
    Args:
        model_path: Path to saved model checkpoint (optional)
        device: Device to load model on ('cpu' or 'cuda')
        pretrained: Whether to use pretrained weights if no checkpoint provided
    
    Returns:
        Loaded model in eval mode
    """
    model = SimpleControlModel(pretrained=pretrained)
    
    if model_path:
        # Load from checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Using {'pretrained' if pretrained else 'randomly initialized'} model")
    
    model.to(device)
    model.eval()
    return model


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for model inference.
    
    Args:
        image: PIL Image, numpy array, or torch tensor
        target_size: Target size for resizing (default: (224, 224))
    
    Returns:
        Preprocessed torch tensor ready for model input
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        # Handle BGR to RGB conversion if it's an OpenCV image
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image = Image.fromarray(image)
    
    # Define ImageNet normalization transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    if isinstance(image, Image.Image):
        tensor = transform(image)
    else:
        tensor = image
    
    # Add batch dimension if missing
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def run_inference(model, image, device='cpu', return_probabilities=True):
    """
    Run inference on a single image frame.
    
    Args:
        model: Loaded model in eval mode
        image: Input image (PIL Image, numpy array, or torch tensor)
        device: Device to run inference on ('cpu' or 'cuda')
        return_probabilities: If True, return probabilities; else return raw logits
    
    Returns:
        Dictionary with:
            - 'prediction': Class index with highest probability
            - 'probabilities': Softmax probabilities for each class
            - 'confidence': Confidence score (max probability)
            - 'logits': Raw model output (if return_probabilities=False)
    """
    # Preprocess image
    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        
        if return_probabilities:
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            return {
                'prediction': prediction.item(),
                'probabilities': probabilities[0].cpu().numpy(),
                'confidence': confidence.item(),
                'logits': logits[0].cpu().numpy()
            }
        else:
            return {
                'logits': logits[0].cpu().numpy()
            }


def get_class_names():
    """
    Get human-readable class names.
    
    Returns:
        List of class names
    """
    # For binary classification: 0 = no_object, 1 = object_detected
    return ['no_object', 'object_detected']


if __name__ == "__main__":
    # Test the model loading and inference
    print("Testing model loading...")
    model = load_model(device='cpu')
    
    print("\nModel architecture:")
    print(model)
    
    # Create a dummy test image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    print("\nRunning inference on test image...")
    result = run_inference(model, test_image, device='cpu')
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")
    
    class_names = get_class_names()
    print(f"Class: {class_names[result['prediction']]}")

