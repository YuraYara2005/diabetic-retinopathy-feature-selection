import torch
import torch.nn as nn
from torchvision import models

class ResNetFeatureExtractor(nn.Module):
    """
    Wraps a pre-trained ResNet50 to extract the final global average pooled 
    features before the classification layer.
    """
    def __init__(self, use_gpu=True):
        super(ResNetFeatureExtractor, self).__init__()
        
        # Determine the device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"Using device: {self.device}")

        # Load pre-trained ResNet50
        # Using the latest weights enum as per current PyTorch standards
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the final fully connected layer (fc)
        # ResNet50 architecture: ... -> Global Average Pool -> FC Layer
        # We keep everything up to the pooling layer.
        self.feature_layers = nn.Sequential(*list(base_model.children())[:-1])
        
        self.to(self.device)
        self.eval()  # Set to evaluation mode (freezes BatchNorm, etc.)

    def forward(self, x):
        """
        Forward pass to extract features.
        Output shape: [Batch, 2048]
        """
        with torch.no_grad():
            x = x.to(self.device)
            features = self.feature_layers(x)
            # Flatten the [Batch, 2048, 1, 1] output to [Batch, 2048]
            return torch.flatten(features, 1)

def get_extractor():
    """Factory function for easy instantiation."""
    return ResNetFeatureExtractor()

if __name__ == "__main__":
    # Quick Test
    extractor = get_extractor()
    # Simulate a batch of 4 images (3 channels, 224x224)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = extractor(dummy_input)
    
    print(f"Feature extraction successful!")
    print(f"Output shape: {output.shape}") # Expected: torch.Size([4, 2048])