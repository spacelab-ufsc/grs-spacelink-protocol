import json
import numpy as np
from datetime import datetime, timezone

with open('golds-ufsc.json') as f:
    descriptor = json.load(f)

# packet_id used to define (packet_name, fields_list)
packet_types = {}
for link in descriptor['links']:
    for pkt_type in link['types']:
        # find the pkt_id field's expected value
        id_field = next(f for f in pkt_type['fields'] if f['id'] == 'pkt_id')
        packet_types[id_field['value']] = pkt_type

def decode_packet(payload: bytes) -> dict:
    pkt = list(payload)
    if len(pkt) == 0:
        return {'error': 'Empty payload'}
    pkt_id = pkt[0]
    if pkt_id not in packet_types:
        return {'error': f'Unknown packet ID: {pkt_id}'}
    pkt_type = packet_types[pkt_id]
    result = {
        'packet_type': pkt_type['name'],
        'received_at': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        'fields': {}
    }
    for field in pkt_type['fields']:
        try:
            value = eval(field['conversion'])
        except Exception as e:
            value = f'parse error: {e}'
        result['fields'][field['id']] = {
            'name':  field['name'],
            'value': value,
            'unit':  field['unit']
        }
    return result
