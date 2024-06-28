from flask import Flask, request, jsonify
import paho.mqtt.client as mqtt
import threading

app = Flask(__name__)
data = []
LINK_MQTT = "1b3616dcaa234d0cb2b32a05363e6cdf.s1.eu.hivemq.cloud"
PORT_MQTT = 8883
topic_kualitas_udara = "data/kualitas_udara"
USERNAME_MQTT = "savaqua"
PASSWORD_MQTT = "Savaqua123"
mqtt_client = mqtt.Client()

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    print(f"Received message '{payload}' on topic '{topic}'")

    if topic == topic_kualitas_udara:
        update_data('kualitas_udara', str(payload))

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(topic_kualitas_udara)

def update_data(key, value):
    if not data:
        new_id = 1
        new_data = {'id': new_id, 'kualitas_udara': None}
        data.append(new_data)
    
    latest_data = data[-1]
    latest_data[key] = value

    if latest_data['kualitas_udara'] is not None:
        new_id = latest_data['id'] + 1
        new_data = {'id': new_id, 'kualitas_udara': None}
        data.append(new_data)

mqtt_client.username_pw_set(USERNAME_MQTT, PASSWORD_MQTT)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

def start_mqtt():
    mqtt_client.tls_set()  # Use default CA certificates
    mqtt_client.connect(LINK_MQTT, PORT_MQTT, 60)
    mqtt_client.loop_forever()

mqtt_thread = threading.Thread(target=start_mqtt)
mqtt_thread.start()

@app.route('/sensor/data', methods=['GET'])
def get_data():
    return jsonify(data)

@app.route('/sensor/data/<int:id>', methods=['GET'])
def get_data_by_id(id):
    result = next((item for item in data if item["id"] == id), None)
    if result:
        return jsonify(result)
    else:
        return jsonify({'pesan': 'Data tidak ditemukan'}), 404

@app.route('/sensor/data', methods=['POST'])
def add_data():
    req_data = request.get_json()
    if not req_data or 'kualitas_udara' not in req_data:
        return jsonify({'pesan': 'Request gagal'}), 400

    new_id = len(data) + 1
    new_data = {
        'id': new_id,
        'kualitas_udara': req_data['kualitas_udara']
    }
    data.append(new_data)
    return jsonify({'pesan': 'Data sukses diterima', 'data': new_data}), 201

if __name__ == '__main__':
    app.run(debug=True, port=6969)
