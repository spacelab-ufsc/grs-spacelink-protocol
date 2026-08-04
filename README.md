# Packet Decoder

Decodes raw telemetry/command packets received by the ground station, using a JSON descriptor (`golds-ufsc.json`) that defines every known packet type and how to parse its fields.

## Overview

The decoder has two stages:

1. **Load time** — `golds-ufsc.json` is parsed once at import. It describes one or more communication `links`, each of which lists the packet `types` it can carry. Every packet type is flattened into a single lookup table, `packet_types`, keyed by the packet's ID byte (the value of its `pkt_id` field).
2. **Runtime** — `decode_packet(payload)` reads the first byte of a raw payload to identify the packet type, then walks that type's field list, evaluating each field's `conversion` expression to extract its value.

## Descriptor format (`golds-ufsc.json`)

The descriptor is expected to have this shape:

```json
{
  "links": [
    {
      "types": [
        {
          "name": "Beacon",
          "fields": [
            { "id": "pkt_id", "value": 1, "name": "Packet ID", "unit": null, "conversion": "pkt[0]" },
            { "id": "voltage", "name": "Battery Voltage", "unit": "V", "conversion": "pkt[1] * 0.01" }
          ]
        }
      ]
    }
  ]
}
```

- **`links[].types[]`** — packet type definitions. Types from all links are merged into one global table, so `pkt_id` values must be unique across the whole descriptor.
- **`pkt_id` field** — every packet type must have exactly one field with `id == "pkt_id"`; its `value` is the fixed byte that identifies this packet type on the wire (not a decoded value).
- **`conversion`** — a Python expression, evaluated with `pkt` (the payload as a `list[int]`) and `numpy` (as `np`) in scope. Used to extract each field's value from the raw bytes, e.g. `"pkt[1] | (pkt[2] << 8)"` or `"np.int16(pkt[3]) * 0.01"`.
- **`name`** / **`unit`** — human-readable metadata carried through to the decoded output.

## API

### `decode_packet(payload: bytes) -> dict`

Decodes one raw packet.

**Success:**

```json
{
  "packet_type": "Beacon",
  "received_at": "2026-07-21T14:03:11.482+00:00",
  "fields": {
    "voltage": { "name": "Battery Voltage", "value": 3.7, "unit": "V" }
  }
}
```

- `packet_type` — the matched type's `name` from the descriptor.
- `received_at` — UTC timestamp (ISO-8601, millisecond precision) of when decoding occurred, not when the packet was transmitted.
- `fields` — one entry per field in the packet type, keyed by field `id`.
- If a field's `conversion` expression raises, that field's `value` is set to a `"parse error: ..."` string instead of failing the whole packet — the rest of the fields still decode.

**Error cases** (returned instead of the structure above):

| Condition | Return value |
|---|---|
| Empty payload | `{"error": "Empty payload"}` |
| First byte doesn't match any known `pkt_id` | `{"error": "Unknown packet ID: <id>"}` |

## Example usage

```python
from decoder import decode_packet

raw = bytes([0x01, 0x8A, 0x01])
result = decode_packet(raw)
print(result['packet_type'], result['fields'])
```

## Security note

Field `conversion` expressions are evaluated with Python's `eval()`. This is safe only because the descriptor is a trusted, version-controlled file maintained by this repo. **Do not point this decoder at a descriptor file from an untrusted or externally supplied source**.

## Known limitations

- All packet types across all links share one global `pkt_id` namespace; adding a new link with a colliding `pkt_id` will silently overwrite an existing entry in `packet_types`.
