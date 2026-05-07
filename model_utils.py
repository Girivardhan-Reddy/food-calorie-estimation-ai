import torch
import torch.nn as nn
from torchvision import models, transforms
import os
from PIL import Image, ImageDraw
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from ultralytics import YOLO

class EnhancedFoodDetector:
    def __init__(self, model_path, data_dir, device='cpu', yolo_model_path='yolov8m.pt'):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.data_dir = data_dir
        self.class_names = self._get_class_names()
        
        # Load EfficientNet model for classification
        self.efficientnet_model = self._load_efficientnet(model_path)
        self.transform = self._get_transform()
        
        # Load YOLO model for object detection
        self.yolo_model = self._load_yolo_model(yolo_model_path)
        
        # Nutrition database
        self.nutrition_db = self._load_nutrition_db()
        
    def _get_class_names(self):
        """Extract class names from train directory"""
        train_path = os.path.join(self.data_dir, 'train')
        if os.path.exists(train_path):
            classes = sorted([d for d in os.listdir(train_path) 
                            if os.path.isdir(os.path.join(train_path, d))])
            return classes
        else:
            return [
                "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
                "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
                "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
                "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
                "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
                "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
                "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
                "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
                "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
                "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
                "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
                "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
                "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
                "mussels", "nachos", "omelette", "onion_rings", "oysters",
                "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
                "pho", "pizza", "pork_chop", "poutine", "prime_rib",
                "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
                "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
                "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
                "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
                "waffles"
            ]
    
    def _load_efficientnet(self, model_path):
        """Load EfficientNet-B0 model"""
        model = models.efficientnet_b0(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(self.class_names))
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}. Using untrained model.")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_yolo_model(self, yolo_model_path):
        """Load YOLOv8 model for object detection"""
        try:
            # Check if custom model exists, otherwise use pretrained
            if os.path.exists(yolo_model_path):
                model = YOLO(yolo_model_path)
                print(f"YOLO model loaded from {yolo_model_path}")
            else:
                # Download and load pretrained YOLOv8m
                model = YOLO('yolov8m.pt')
                print("YOLOv8m pretrained model loaded")
            return model
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            # Fallback to None - will use full image classification
            return None
    
    def _get_transform(self):
        """Image preprocessing transformations for EfficientNet"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_nutrition_db(self):
        """Create nutrition database"""
        nutrition_db = {
            "apple_pie": {"calories_per_100g": 265},
            "baby_back_ribs": {"calories_per_100g": 380},
            "baklava": {"calories_per_100g": 430},
            "beef_carpaccio": {"calories_per_100g": 250},
            "beef_tartare": {"calories_per_100g": 230},
            "beet_salad": {"calories_per_100g": 90},
            "beignets": {"calories_per_100g": 420},
            "bibimbap": {"calories_per_100g": 180},
            "bread_pudding": {"calories_per_100g": 280},
            "breakfast_burrito": {"calories_per_100g": 220},
            "bruschetta": {"calories_per_100g": 180},
            "caesar_salad": {"calories_per_100g": 150},
            "cannoli": {"calories_per_100g": 350},
            "caprese_salad": {"calories_per_100g": 200},
            "carrot_cake": {"calories_per_100g": 415},
            "cheesecake": {"calories_per_100g": 321},
            "chicken_curry": {"calories_per_100g": 165},
            "chicken_wings": {"calories_per_100g": 290},
            "chocolate_cake": {"calories_per_100g": 371},
            "chocolate_mousse": {"calories_per_100g": 225},
            "churros": {"calories_per_100g": 470},
            "clam_chowder": {"calories_per_100g": 120},
            "club_sandwich": {"calories_per_100g": 250},
            "crab_cakes": {"calories_per_100g": 185},
            "creme_brulee": {"calories_per_100g": 265},
            "croque_madame": {"calories_per_100g": 320},
            "cup_cakes": {"calories_per_100g": 305},
            "donuts": {"calories_per_100g": 452},
            "dumplings": {"calories_per_100g": 190},
            "edamame": {"calories_per_100g": 122},
            "eggs_benedict": {"calories_per_100g": 287},
            "escargots": {"calories_per_100g": 210},
            "falafel": {"calories_per_100g": 333},
            "filet_mignon": {"calories_per_100g": 335},
            "fish_and_chips": {"calories_per_100g": 290},
            "foie_gras": {"calories_per_100g": 462},
            "french_fries": {"calories_per_100g": 312},
            "french_onion_soup": {"calories_per_100g": 140},
            "french_toast": {"calories_per_100g": 229},
            "fried_calamari": {"calories_per_100g": 190},
            "fried_rice": {"calories_per_100g": 168},
            "frozen_yogurt": {"calories_per_100g": 159},
            "garlic_bread": {"calories_per_100g": 350},
            "gnocchi": {"calories_per_100g": 210},
            "greek_salad": {"calories_per_100g": 100},
            "grilled_cheese_sandwich": {"calories_per_100g": 350},
            "grilled_salmon": {"calories_per_100g": 206},
            "guacamole": {"calories_per_100g": 160},
            "gyoza": {"calories_per_100g": 195},
            "hamburger": {"calories_per_100g": 295},
            "hot_and_sour_soup": {"calories_per_100g": 80},
            "hot_dog": {"calories_per_100g": 290},
            "huevos_rancheros": {"calories_per_100g": 195},
            "hummus": {"calories_per_100g": 166},
            "ice_cream": {"calories_per_100g": 207},
            "lasagna": {"calories_per_100g": 135},
            "lobster_bisque": {"calories_per_100g": 145},
            "lobster_roll_sandwich": {"calories_per_100g": 320},
            "macaroni_and_cheese": {"calories_per_100g": 164},
            "macarons": {"calories_per_100g": 405},
            "miso_soup": {"calories_per_100g": 35},
            "mussels": {"calories_per_100g": 172},
            "nachos": {"calories_per_100g": 350},
            "omelette": {"calories_per_100g": 154},
            "onion_rings": {"calories_per_100g": 385},
            "oysters": {"calories_per_100g": 81},
            "pad_thai": {"calories_per_100g": 153},
            "paella": {"calories_per_100g": 180},
            "pancakes": {"calories_per_100g": 227},
            "panna_cotta": {"calories_per_100g": 280},
            "peking_duck": {"calories_per_100g": 337},
            "pho": {"calories_per_100g": 85},
            "pizza": {"calories_per_100g": 266},
            "pork_chop": {"calories_per_100g": 231},
            "poutine": {"calories_per_100g": 315},
            "prime_rib": {"calories_per_100g": 355},
            "pulled_pork_sandwich": {"calories_per_100g": 280},
            "ramen": {"calories_per_100g": 180},
            "ravioli": {"calories_per_100g": 203},
            "red_velvet_cake": {"calories_per_100g": 367},
            "risotto": {"calories_per_100g": 166},
            "samosa": {"calories_per_100g": 262},
            "sashimi": {"calories_per_100g": 150},
            "scallops": {"calories_per_100g": 111},
            "seaweed_salad": {"calories_per_100g": 45},
            "shrimp_and_grits": {"calories_per_100g": 155},
            "spaghetti_bolognese": {"calories_per_100g": 130},
            "spaghetti_carbonara": {"calories_per_100g": 185},
            "spring_rolls": {"calories_per_100g": 200},
            "steak": {"calories_per_100g": 271},
            "strawberry_shortcake": {"calories_per_100g": 346},
            "sushi": {"calories_per_100g": 150},
            "tacos": {"calories_per_100g": 226},
            "takoyaki": {"calories_per_100g": 180},
            "tiramisu": {"calories_per_100g": 310},
            "tuna_tartare": {"calories_per_100g": 185},
            "waffles": {"calories_per_100g": 291}
        }
        
        avg_calories = 250
        for class_name in self.class_names:
            if class_name not in nutrition_db:
                nutrition_db[class_name] = {"calories_per_100g": avg_calories}
        
        return nutrition_db
    
    def classify_crop(self, crop_image):
        """Classify a single cropped image using EfficientNet"""
        # Transform for model
        input_tensor = self.transform(crop_image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.efficientnet_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, 1)
        
        # Get top prediction
        class_idx = top_indices[0].item()
        class_name = self.class_names[class_idx]
        confidence = top_probs[0].item() * 100
        
        return class_name, confidence
    
    def draw_bounding_box(self, image, bbox, food_name=None, confidence=None):
        """Draw enhanced bounding box with label"""
        draw = ImageDraw.Draw(image)
        
        # Bounding box coordinates
        (x1, y1), (x2, y2) = bbox
        
        # Draw rectangle with shadow effect
        shadow_offset = 2
        draw.rectangle([x1 + shadow_offset, y1 + shadow_offset, 
                       x2 + shadow_offset, y2 + shadow_offset], 
                     outline="rgba(0,0,0,150)", width=3)
        
        # Main rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Create label text
        if food_name and confidence:
            label = f"{food_name} ({confidence:.1f}%)"
        else:
            label = "Detected Food"
        
        # Calculate text size
        try:
            from PIL import ImageFont
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
        except:
            font = None
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 30), label, font=font)
        draw.rectangle(text_bbox, fill="red")
        
        # Draw label text
        draw.text((x1, y1 - 30), label, fill="white", font=font)
        
        # Draw corners for better visibility
        corner_length = 15
        # Top-left
        draw.line([x1, y1, x1 + corner_length, y1], fill="red", width=3)
        draw.line([x1, y1, x1, y1 + corner_length], fill="red", width=3)
        # Top-right
        draw.line([x2, y1, x2 - corner_length, y1], fill="red", width=3)
        draw.line([x2, y1, x2, y1 + corner_length], fill="red", width=3)
        # Bottom-left
        draw.line([x1, y2, x1 + corner_length, y2], fill="red", width=3)
        draw.line([x1, y2, x1, y2 - corner_length], fill="red", width=3)
        # Bottom-right
        draw.line([x2, y2, x2 - corner_length, y2], fill="red", width=3)
        draw.line([x2, y2, x2, y2 - corner_length], fill="red", width=3)
        
        return image
    
    def estimate_portion_size(self, image, bbox):
        """Estimate portion size based on bbox area"""
        img_width, img_height = image.size
        (x1, y1), (x2, y2) = bbox
        
        # Calculate area ratio
        food_area = (x2 - x1) * (y2 - y1)
        total_area = img_width * img_height
        food_ratio = food_area / total_area
        
        # Apply logistic function for better estimation
        import math
        food_ratio = 1 / (1 + math.exp(-10 * (food_ratio - 0.5)))
        
        # Cap the ratio between 0.1 and 0.9
        food_ratio = max(0.1, min(food_ratio, 0.9))
        
        return food_ratio
    
    def estimate_weight(self, food_class, portion_ratio):
        """Enhanced weight estimation with more categories"""
        reference_weights = {
            "pizza": 300, "burger": 250, "steak": 200, "pasta": 250,
            "salad": 150, "sandwich": 200, "soup": 300, "dessert": 150,
            "rice": 200, "seafood": 180, "chicken": 200, "vegetables": 150,
            "bread": 100, "egg": 150, "noodles": 200, "cheese": 100,
            "fruit": 150, "cereal": 100, "fish": 180, "pork": 200,
            "beef": 200, "lamb": 180, "duck": 200, "shrimp": 150,
            "sausage": 150, "bacon": 100, "tofu": 150, "bean": 150,
            "nut": 50, "yogurt": 150, "milk": 250, "juice": 250,
            "coffee": 250, "tea": 250, "smoothie": 300, "shake": 300
        }
        
        food_categories = {
            # Pizza and similar
            "pizza": "pizza", "calzone": "pizza", "quiche": "pizza",
            
            # Burgers and sandwiches
            "hamburger": "burger", "cheeseburger": "burger", 
            "club_sandwich": "sandwich", "grilled_cheese_sandwich": "sandwich",
            "croque_madame": "sandwich", "lobster_roll_sandwich": "sandwich",
            "pulled_pork_sandwich": "sandwich", "breakfast_burrito": "sandwich",
            
            # Steaks and meats
            "steak": "steak", "filet_mignon": "steak", "prime_rib": "steak",
            
            # Pasta
            "spaghetti_bolognese": "pasta", "spaghetti_carbonara": "pasta",
            "lasagna": "pasta", "ravioli": "pasta", "gnocchi": "pasta",
            "macaroni_and_cheese": "pasta",
            
            # Salads
            "caesar_salad": "salad", "greek_salad": "salad", 
            "beet_salad": "salad", "seaweed_salad": "salad",
            "caprese_salad": "salad",
            
            # Soups
            "french_onion_soup": "soup", "clam_chowder": "soup",
            "lobster_bisque": "soup", "miso_soup": "soup", 
            "pho": "soup", "hot_and_sour_soup": "soup",
            
            # Desserts
            "cheesecake": "dessert", "chocolate_cake": "dessert",
            "apple_pie": "dessert", "carrot_cake": "dessert",
            "red_velvet_cake": "dessert", "tiramisu": "dessert",
            "creme_brulee": "dessert", "panna_cotta": "dessert",
            "chocolate_mousse": "dessert", "strawberry_shortcake": "dessert",
            "bread_pudding": "dessert", "cannoli": "dessert",
            "cup_cakes": "dessert", "macarons": "dessert",
            "beignets": "dessert", "churros": "dessert",
            "donuts": "dessert", "frozen_yogurt": "dessert",
            "ice_cream": "dessert",
            
            # Rice dishes
            "fried_rice": "rice", "risotto": "rice", "paella": "rice",
            "bibimbap": "rice",
            
            # Seafood
            "fish_and_chips": "seafood", "grilled_salmon": "seafood",
            "crab_cakes": "seafood", "fried_calamari": "seafood",
            "mussels": "seafood", "oysters": "seafood",
            "scallops": "seafood", "sushi": "seafood",
            "sashimi": "seafood", "ceviche": "seafood",
            "tuna_tartare": "seafood",
            
            # Chicken and poultry
            "chicken_curry": "chicken", "chicken_wings": "chicken",
            "chicken_quesadilla": "chicken",
            
            # Pork
            "baby_back_ribs": "pork", "pork_chop": "pork",
            "pulled_pork_sandwich": "pork",
            
            # Vegetables and sides
            "edamame": "vegetables", "guacamole": "vegetables",
            "hummus": "vegetables", "french_fries": "vegetables",
            "onion_rings": "vegetables", "spring_rolls": "vegetables",
            "samosa": "vegetables", "gyoza": "vegetables",
            "dumplings": "vegetables", "falafel": "vegetables",
            "nachos": "vegetables", "tacos": "vegetables",
            "takoyaki": "vegetables", "bruschetta": "vegetables",
            "garlic_bread": "bread",
            
            # Egg dishes
            "omelette": "egg", "huevos_rancheros": "egg",
            "eggs_benedict": "egg", "deviled_eggs": "egg",
            
            # Breakfast
            "pancakes": "bread", "french_toast": "bread",
            "waffles": "bread",
            
            # Noodles
            "pad_thai": "noodles", "ramen": "noodles",
            
            # Default category
            "default": "default"
        }
        
        # Get category or use default
        category = food_categories.get(food_class, "default")
        
        # Get base weight
        if category == "default":
            # Estimate based on common serving sizes
            if "soup" in food_class or "stew" in food_class:
                base_weight = 300
            elif "salad" in food_class:
                base_weight = 200
            elif "cake" in food_class or "pie" in food_class:
                base_weight = 150
            else:
                base_weight = 200
        else:
            base_weight = reference_weights.get(category, 200)
        
        # Adjust weight based on portion ratio
        if portion_ratio > 0.7:
            weight = base_weight * 1.2
        elif portion_ratio > 0.4:
            weight = base_weight
        elif portion_ratio > 0.2:
            weight = base_weight * 0.7
        else:
            weight = base_weight * 0.5
        
        # Add random variation (±10%) for realism
        import random
        variation = 1 + random.uniform(-0.1, 0.1)
        weight *= variation
        
        # Ensure weight is within reasonable bounds
        return max(30, min(weight, 1000))
    
    def predict(self, image_path):
        """Make prediction on uploaded image - detects multiple food items"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            
            detected_items = []
            img_width, img_height = image.size
            
            # Run YOLO detection if model is available
            if self.yolo_model is not None:
                # Run YOLO inference
                results = self.yolo_model(image_path)
                
                # Process detections
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    for box in boxes:
                        # Get confidence
                        confidence = float(box.conf[0]) * 100
                        
                        # Only consider detections with confidence > 40%
                        if confidence < 40:
                            continue
                        
                        # Get bbox coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_width, x2)
                        y2 = min(img_height, y2)
                        
                        # Skip if bbox is too small
                        if (x2 - x1) < 50 or (y2 - y1) < 50:
                            continue
                        
                        # Crop the detected region
                        crop_image = image.crop((x1, y1, x2, y2))
                        
                        # Classify using EfficientNet
                        food_class, class_confidence = self.classify_crop(crop_image)
                        
                        # Use the higher confidence between YOLO and classification
                        final_confidence = max(confidence, class_confidence)
                        
                        # Create bbox tuple
                        bbox = [(x1, y1), (x2, y2)]
                        
                        # Estimate portion size using bbox area
                        portion_ratio = self.estimate_portion_size(image, bbox)
                        
                        # Estimate weight
                        estimated_weight = self.estimate_weight(food_class, portion_ratio)
                        
                        # Get calories from nutrition database
                        if food_class in self.nutrition_db:
                            calories_per_100g = self.nutrition_db[food_class]["calories_per_100g"]
                        else:
                            calories_per_100g = 250
                        
                        # Calculate estimated calories
                        estimated_calories = (estimated_weight / 100) * calories_per_100g
                        
                        # Add to detected items
                        detected_items.append({
                            "food_name": food_class.replace('_', ' ').title(),
                            "confidence": round(final_confidence, 2),
                            "estimated_weight": round(estimated_weight, 1),
                            "estimated_calories": round(estimated_calories, 1),
                            "portion_ratio": round(portion_ratio * 100, 1),
                            "calories_per_100g": calories_per_100g,
                            "bbox": bbox
                        })
            
            # Fallback: If no items detected or YOLO not available, use full image classification
            if not detected_items:
                # Full image classification
                food_class, confidence = self.classify_crop(image)
                
                # Estimate portion (full image = 0.5 ratio)
                portion_ratio = 0.5
                
                # Estimate weight
                estimated_weight = self.estimate_weight(food_class, portion_ratio)
                
                # Get calories from nutrition database
                if food_class in self.nutrition_db:
                    calories_per_100g = self.nutrition_db[food_class]["calories_per_100g"]
                else:
                    calories_per_100g = 250
                
                # Calculate estimated calories
                estimated_calories = (estimated_weight / 100) * calories_per_100g
                
                # Create bbox for full image
                bbox = [(0, 0), (img_width, img_height)]
                
                detected_items.append({
                    "food_name": food_class.replace('_', ' ').title(),
                    "confidence": round(confidence, 2),
                    "estimated_weight": round(estimated_weight, 1),
                    "estimated_calories": round(estimated_calories, 1),
                    "portion_ratio": round(portion_ratio * 100, 1),
                    "calories_per_100g": calories_per_100g,
                    "bbox": bbox
                })
            
            # Draw bounding boxes for all detected items
            image_with_bbox = original_image.copy()
            for item in detected_items:
                image_with_bbox = self.draw_bounding_box(
                    image_with_bbox,
                    item["bbox"],
                    item["food_name"],
                    item["confidence"]
                )
            
            # Calculate total calories
            total_calories = sum(item["estimated_calories"] for item in detected_items)
            
            # Prepare final output
            output = {
                "results": detected_items,
                "image_with_bbox": image_with_bbox,
                "total_calories": round(total_calories, 1)
            }
            
            return output
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


class FoodCalorieEstimator(EnhancedFoodDetector):
    """Compatibility class"""
    pass