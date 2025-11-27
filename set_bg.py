import cv2
import numpy as np

car = cv2.imread("static/results/c7b83325-5ce2-43e6-956b-e5f3e8a0d884.png", cv2.IMREAD_UNCHANGED)
mask = cv2.imread("static/masks/c7b83325-5ce2-43e6-956b-e5f3e8a0d884.png", cv2.IMREAD_GRAYSCALE)
bg = cv2.imread("static/custom-bg3.png")

# Handle 4-channel input
if car.shape[2] == 4:
    car_rgb = car[:, :, :3]
else:
    car_rgb = car

# --- NEW: RESIZE CAR TO FIT BACKGROUND ---

bg_h, bg_w = bg.shape[:2]
car_h, car_w = car_rgb.shape[:2]

# Calculate scale to fit car into background (e.g., occupy 80% of width or height)
scale_width = (bg_w * 0.8) / car_w
scale_height = (bg_h * 0.8) / car_h
scale = min(scale_width, scale_height) # Use min to preserve aspect ratio

new_w = int(car_w * scale)
new_h = int(car_h * scale)

# Resize the car and the original mask
car_resized = cv2.resize(car_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

# PLACE CAR ON BACKGROUND CANVAS ---

# Create empty canvas matching Background size
car_canvas = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
mask_canvas = np.zeros((bg_h, bg_w), dtype=np.uint8)

# Calculate centering coordinates adn center horizontally
x_offset = (bg_w - new_w) // 2
# Place near bottom (e.g., 10% padding from bottom)
y_offset = int(bg_h * 0.9) - new_h 

# Place the resized car onto the canvas
car_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = car_resized
mask_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_resized

# --- PREPARE MASKS (Using the new Canvas Mask) ---

mask_float = mask_canvas.astype(np.float32) / 255.0

# Cleanup edges
kernel_erode = np.ones((3, 3), np.uint8)
mask_eroded = cv2.erode(mask_canvas, kernel_erode, iterations=1)
mask_eroded_float = mask_eroded.astype(np.float32) / 255.0
mask_soft = cv2.GaussianBlur(mask_eroded_float, (3, 3), 0)
mask_3 = np.dstack([mask_soft, mask_soft, mask_soft])

# --- FIND CAR "FEET" ON THE CANVAS ---
y_nonzero, x_nonzero = np.nonzero(mask_canvas)

if len(y_nonzero) > 0:
    y_max = np.max(y_nonzero) 
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero)
    car_width = x_max - x_min
    car_center_x = x_min + (car_width // 2)
else:
    # Fallback if mask is empty
    y_max = bg_h - 1
    car_center_x = bg_w // 2

# --- GENERATE SHADOWS

# 1. CONTACT SHADOW
contact_scale_y = 0.08 
contact_width_scale = 0.90
M_contact = np.float32([
    [contact_width_scale, 0, car_center_x * (1 - contact_width_scale)], 
    [0, contact_scale_y, y_max - (y_max * contact_scale_y)] 
])

contact_shadow = cv2.warpAffine(mask_canvas, M_contact, (bg_w, bg_h))
contact_shadow = cv2.GaussianBlur(contact_shadow, (9, 5), 0) 
contact_shadow = contact_shadow.astype(np.float32) / 255.0

# 2. AMBIENT SHADOW
ambient_scale_y = 0.25 
M_ambient = np.float32([
    [1, 0, 0], 
    [0, ambient_scale_y, y_max - (y_max * ambient_scale_y)]
])

ambient_shadow = cv2.warpAffine(mask_canvas, M_ambient, (bg_w, bg_h))
ambient_shadow = cv2.GaussianBlur(ambient_shadow, (75, 45), 0) # Heavier blur for larger BG
ambient_shadow = ambient_shadow.astype(np.float32) / 255.0

# Combine
combined_shadow = np.maximum(contact_shadow * 0.15, ambient_shadow * 0.5)
shadow_3 = np.dstack([combined_shadow, combined_shadow, combined_shadow])

# --- COMPOSITING ---

bg_float = bg.astype(np.float32) / 255.0 # Use original BG
bg_shadowed = bg_float * (1.0 - shadow_3)

car_float = car_canvas.astype(np.float32) / 255.0

foreground = car_float * mask_3
background_masked = bg_shadowed * (1.0 - mask_3)

final = foreground + background_masked

# --- Post-Processing ---
final = np.clip(final * 255, 0, 255).astype(np.uint8)

cv2.imwrite("output_fixed_bg.png", final)
print(f"Saved output_fixed_bg.png with size {final.shape}")