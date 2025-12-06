#!/usr/bin/env python3
"""
Script to apply the ResNet ensemble fix to NewVerPynbAgent.ipynb
This adds ResNetDocumentClassifier and updates EnsembleDocumentClassifier.load_models()
"""

import json
import sys

def apply_fix(notebook_path):
    """Apply the ResNet ensemble fix to the notebook"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # ResNetDocumentClassifier class code
    resnet_classifier_code = '''class ResNetDocumentClassifier:
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
'''
    
    # Updated load_models code
    updated_load_models = '''            # Check if this is a ResNet model
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
                classifier.load_model()'''
    
    updated_processor_code = '''            # Use first classifier's processor/transform for all
            if self.processor is None:
                if hasattr(classifier, 'processor'):
                    self.processor = classifier.processor
                elif hasattr(classifier, 'transform'):
                    self.processor = classifier.transform'''
    
    # Find and modify cells
    modified = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Add ResNetDocumentClassifier after DocumentClassifier
            if 'class EnsembleDocumentClassifier:' in source and 'class ResNetDocumentClassifier:' not in source:
                # Find where to insert (after DocumentClassifier, before EnsembleDocumentClassifier)
                if 'print(f"Model loaded from: {path}")' in source and 'class ResNetDocumentClassifier:' not in source:
                    # Replace the pattern: print(...) followed by blank lines and then EnsembleDocumentClassifier
                    pattern = 'print(f"Model loaded from: {path}")\n\n\nclass EnsembleDocumentClassifier:'
                    replacement = f'print(f"Model loaded from: {{path}}")\n\n\n{resnet_classifier_code}\n\n\nclass EnsembleDocumentClassifier:'
                    
                    if pattern in source:
                        new_source = source.replace(pattern, replacement)
                        cell['source'] = new_source.splitlines(keepends=True)
                        if cell['source'] and not cell['source'][-1].endswith('\n'):
                            cell['source'][-1] += '\n'
                        modified = True
                        print("✓ Added ResNetDocumentClassifier class")
                    else:
                        # Try alternative pattern matching
                        lines = source.splitlines()
                        new_lines = []
                        inserted = False
                        for i, line in enumerate(lines):
                            new_lines.append(line)
                            # Look for the transition point
                            if 'class EnsembleDocumentClassifier:' in line and not inserted:
                                # Check if previous lines suggest DocumentClassifier ended
                                if i > 2 and 'print(f"Model loaded from:' in '\n'.join(lines[max(0,i-5):i]):
                                    # Insert ResNetDocumentClassifier before this line
                                    resnet_lines = resnet_classifier_code.splitlines()
                                    for resnet_line in resnet_lines:
                                        new_lines.insert(-1, resnet_line)
                                    new_lines.insert(-1, '')  # Empty line
                                    new_lines.insert(-1, '')  # Another empty line
                                    inserted = True
                        
                        if inserted:
                            cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_lines]
                            # Ensure last line doesn't have extra newline
                            if cell['source'] and cell['source'][-1].endswith('\n\n'):
                                cell['source'][-1] = cell['source'][-1].rstrip('\n') + '\n'
                            modified = True
                            print("✓ Added ResNetDocumentClassifier class")
            
            # Update load_models method
            if 'def load_models(self):' in source and 'is_resnet =' not in source:
                # Replace the DocumentClassifier instantiation
                old_pattern = "classifier = DocumentClassifier(\n                num_labels=2,\n                model_path=cfg['path']\n            )\n            classifier.load_model()"
                new_source = source.replace(
                    "classifier = DocumentClassifier(\n                num_labels=2,\n                model_path=cfg['path']\n            )\n            classifier.load_model()",
                    updated_load_models
                )
                
                # Update processor assignment
                old_processor = "# Use first classifier's processor for all\n            if self.processor is None:\n                self.processor = classifier.processor"
                new_processor = updated_processor_code
                
                if old_processor in new_source:
                    new_source = new_source.replace(old_processor, new_processor)
                
                if new_source != source:
                    cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in new_source.split('\n')]
                    # Remove trailing newline from last line
                    if cell['source'] and cell['source'][-1] == '\n':
                        cell['source'][-1] = cell['source'][-1].rstrip('\n')
                    modified = True
                    print("✓ Updated load_models() method")
    
    if modified:
        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"\n✅ Successfully applied fixes to {notebook_path}")
        return True
    else:
        print("⚠️  No changes were made. The fix may have already been applied.")
        return False

if __name__ == '__main__':
    notebook_path = 'NewVerPynbAgent.ipynb'
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
    
    try:
        apply_fix(notebook_path)
    except Exception as e:
        print(f"❌ Error applying fix: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

