# ResNet Ensemble Fix

## Problem
The `EnsembleDocumentClassifier` was creating `DocumentClassifier` (ViT-based) instances for ALL configs, including "resnet18". This caused:
- Shape mismatch when trying to load ResNet18 weights into ViT model
- Exception was caught silently, so ResNet weights were never used
- Ensemble ended up being all ViTs, zero ResNet

## Solution

### 1. Create ResNetDocumentClassifier Class

Add this class right after the `DocumentClassifier` class definition:

```python
class ResNetDocumentClassifier:
    """Uses ResNet18 to classify docs as receipt or not."""
    
    def __init__(self, num_labels=2, model_path=None):
        self.num_labels = num_labels
        self.model = None
        self.model_path = model_path or os.path.join(MODELS_DIR, 'rvl_resnet18.pt')
        self.use_class_mapping = False
        
        # ResNet18 uses standard ImageNet transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_model(self):
        """Load ResNet18 architecture - will determine num_classes from checkpoint"""
        from torchvision import models
        import torch.nn as nn
        
        # Create ResNet18 model - we'll modify final layer after loading weights
        self.model = models.resnet18(weights=None)
        self.model = self.model.to(DEVICE)
        self.model.eval()
        return self.model
    
    def load_weights(self, path):
        """Load model weights from checkpoint - handles both 16-class and 2-class models"""
        if self.model is None:
            self.load_model()
        
        from torchvision import models
        import torch.nn as nn
        
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            
            # Extract state dict
            state_dict = None
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            if state_dict is None:
                raise ValueError("Could not extract state dict from checkpoint")
            
            # Determine number of classes from checkpoint
            fc_key = None
            for key in ['fc.weight', 'resnet.fc.weight']:
                if key in state_dict:
                    fc_key = key
                    break
            
            if fc_key is None:
                raise ValueError("Could not find final layer weights in checkpoint")
            
            num_classes_in_checkpoint = state_dict[fc_key].shape[0]
            
            # Create model with correct number of classes
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes_in_checkpoint)
            self.model = self.model.to(DEVICE)
            
            # Load weights
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # Track if we need class mapping (16-class -> receipt/other)
            self.use_class_mapping = (num_classes_in_checkpoint == 16)
            self.num_classes_in_checkpoint = num_classes_in_checkpoint
            
            print(f"Model loaded from: {path} ({num_classes_in_checkpoint} classes)")
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise
    
    def predict(self, image):
        """Check if an image is a receipt - same interface as DocumentClassifier"""
        self.model.eval()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
            # Handle 16-class (RVL-CDIP) vs 2-class models
            if self.use_class_mapping:
                # Map RVL-CDIP classes to receipt/other
                # Class 11 = 'invoice', Class 10 = 'budget' - these are receipt-like
                receipt_like_classes = [11]  # invoice
                receipt_prob = probs[receipt_like_classes].sum().item()
            else:
                # 2-class model: class 1 = receipt, class 0 = other
                receipt_prob = probs[1].item() if self.num_classes_in_checkpoint == 2 else 0.5
        
        return {
            'is_receipt': receipt_prob > 0.5,
            'confidence': receipt_prob,
            'label': 'receipt' if receipt_prob > 0.5 else 'other'
        }
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")
```

### 2. Update EnsembleDocumentClassifier.load_models()

Find the `load_models()` method in `EnsembleDocumentClassifier` and replace it with:

```python
def load_models(self):
    """Load all models in the ensemble"""
    print(f"Loading ensemble of {len(self.model_configs)} models...")
    
    for i, cfg in enumerate(self.model_configs):
        print(f"  [{i+1}/{len(self.model_configs)}] Loading {cfg['name']}...")
        
        # Check if this is a ResNet model
        is_resnet = 'resnet' in cfg['name'].lower()
        
        if is_resnet:
            # Use ResNetDocumentClassifier for ResNet models
            classifier = ResNetDocumentClassifier(
                num_labels=2,
                model_path=cfg['path']
            )
            classifier.load_model()
        else:
            # Use DocumentClassifier for ViT models
            classifier = DocumentClassifier(
                num_labels=2,
                model_path=cfg['path']
            )
            classifier.load_model()
        
        if os.path.exists(cfg['path']):
            try:
                classifier.load_weights(cfg['path'])
            except Exception as e:
                print(f"    Warning: Could not load weights for {cfg['name']}: {e}")
        
        self.classifiers.append(classifier)
        
        # Use first classifier's processor/transform for all
        if self.processor is None:
            if hasattr(classifier, 'processor'):
                self.processor = classifier.processor
            elif hasattr(classifier, 'transform'):
                self.processor = classifier.transform
    
    print(f"Ensemble loaded with {len(self.classifiers)} models")
    print(f"Model weights: {dict(zip([c['name'] for c in self.model_configs], self.weights))}")
    return self
```

## Summary

The fix:
1. ✅ Adds `ResNetDocumentClassifier` class that uses `torchvision.models.resnet18`
2. ✅ Updates `EnsembleDocumentClassifier.load_models()` to check config name and use the right classifier
3. ✅ Both classifiers expose the same `predict(image)` interface returning `is_receipt`, `confidence`, `label`

Now the ensemble will properly load ResNet18 models instead of trying to load them into ViT!

