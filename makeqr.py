import qrcode
import json  # Import the json module

# Define product details
product_data = {
    "name": "Milk",
    "price": "$3.50",
    "description": "1 liter of fresh milk."
}

# Convert product data to a JSON string
product_data_json = json.dumps(product_data)

# Create a QR code from the product data
qr = qrcode.make(product_data_json)

# Save the generated QR code as an image file
qr.save("milk_qr.png")

print("QR Code saved as 'milk_qr.png'.")
