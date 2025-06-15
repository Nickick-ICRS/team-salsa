import json
import uuid
import threading
import time
import websocket
from datetime import datetime

# Replace this ACCID value with your robot's actual serial number (SN)
ACCID = "SF_TRON1A_154"

# Atomic flag for graceful exit
should_exit = False

# WebSocket client instance
ws_client = None

# Generate dynamic GUID
def generate_guid():
    return str(uuid.uuid4())

# Send WebSocket request with title and data
def send_request(title, data=None):
    if data is None:
        data = {}
    
    # Create message structure with necessary fields
    message = {
        "accid": ACCID,
        "title": title,
        "timestamp": int(time.time() * 1000),  # Current timestamp in milliseconds
        "guid": generate_guid(),
        "data": data
    }

    message_str = json.dumps(message)
    
    # Send the message through WebSocket if client is connected
    if ws_client:
        ws_client.send(message_str)

# Handle user commands
def handle_commands():
    global should_exit
    last_command = ""
    while not should_exit:
        command = input("Enter command ('stand', 'walk', 'twist', 'sit', 'stair', 'stop', 'imu', 'recover', 'salsa') or 'exit' to quit:\n")
        
        if command == "exit":
            should_exit = True  # Set exit flag to stop the loop
            break
        elif command == "stand":
            send_request("request_stand_mode")  # Send stand mode request
        elif command == "walk":
            send_request("request_walk_mode")  # Send walk mode request
        elif command == "twist":
            # Get twist values from user
            x = float(input("Enter x value:"))
            y = float(input("Enter y value:"))
            z = float(input("Enter z value:"))
            send_request("request_twist", {"x": x, "y": y, "z": z})
        elif command == "salsa":
            for _ in range(1):
                send_request("request_twist", {"x": 1.0, "y": 0.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 0})
                time.sleep(0.5)
                send_request("request_twist", {"x": -1.0, "y": 0.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 5})
                time.sleep(0.5)
                send_request("request_light_effect", {"effect": 0})
                time.sleep(0.5)
                send_request("request_twist", {"x": -1.0, "y": 0.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 5})
                time.sleep(0.5)
                send_request("request_twist", {"x": 1.0, "y": 0.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 0})
                time.sleep(0.5)
                send_request("request_twist", {"x": 0.0, "y": 1.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 1})
                time.sleep(0.5)
                send_request("request_twist", {"x": 0.0, "y": -1.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 3})
                time.sleep(0.5)
                send_request("request_light_effect", {"effect": 1})
                time.sleep(0.5)
                send_request("request_twist", {"x": 0.0, "y": -1.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 3})
                time.sleep(2.0)
                send_request("request_twist", {"x": 0.0, "y": 1.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 1})
                time.sleep(0.5)
                send_request("request_twist", {"x": 1.0, "y": 0.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 2})
                time.sleep(0.5)
                send_request("request_twist", {"x": -1.0, "y": 0.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 4})
                time.sleep(0.5)
                send_request("request_light_effect", {"effect": 2})
                time.sleep(0.5)
                send_request("request_twist", {"x": -1.0, "y": 0.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 4})
                time.sleep(0.5)
                send_request("request_twist", {"x": 1.0, "y": 0.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 2})
                time.sleep(0.5)
                send_request("request_twist", {"x": 0.0, "y": 1.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 4})
                time.sleep(0.5)
                send_request("request_twist", {"x": 0.0, "y": -1.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 0})
                time.sleep(0.5)
                send_request("request_light_effect", {"effect": 1})
                time.sleep(0.5)
                send_request("request_twist", {"x": 0.0, "y": -1.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 2})
                time.sleep(0.5)
                send_request("request_twist", {"x": 0.0, "y": 1.0, "z": 0.0})
                send_request("request_light_effect", {"effect": 3})
                time.sleep(0.5)
        elif command == "sit":
            send_request("request_sitdown")  # Send sit down request
        elif command == "stair":
            # Get stair mode enable flag from user
            enable = input("Enable stair mode (true/false):").strip().lower() == 'true'
            send_request("request_stair_mode", {"enable": enable})
        elif command == "stop":
            print("Are you sure? 'yes' if so")
        elif last_command == "stop" and command == "yes":
            send_request("request_emgy_stop")  # Send emergency stop request
        elif command == "imu":
            # Get IMU enable flag from user
            enable = input("Enable IMU (true/false):").strip().lower() == 'true'
            send_request("request_enable_imu", {"enable": enable})
        elif command == "recover":
            send_request("request_recover")
        last_command = command

# WebSocket on_open callback
def on_open(ws):
    print("Connected!")
    # Start handling commands in a separate thread
    threading.Thread(target=handle_commands, daemon=True).start()

# WebSocket on_message callback
def on_message(ws, message):
    if "notify_robot_info" in message:
        return
    print(f"Received message: {message}")  # Print the received message

# WebSocket on_close callback
def on_close(ws, close_status_code, close_msg):
    print("Connection closed.")

# Close WebSocket connection
def close_connection(ws):
    ws.close()

def main():
    global ws_client
    
    # Create WebSocket client instance
    ws_client = websocket.WebSocketApp(
        "ws://10.192.1.2:5000",  # WebSocket server URI
        on_open=on_open,
        on_message=on_message,
        on_close=on_close
    )
    
    # Run WebSocket client loop
    print("Press Ctrl+C to exit.")
    ws_client.run_forever()

if __name__ == "__main__":
    main()