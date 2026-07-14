import json
import unittest
import numpy as np
from datetime import datetime, timezone
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spacelink import decode_packet
from spacelink import decode_packet

# ─────────────────────────────────────────────
# Network layer under test (copied from main program)
# ─────────────────────────────────────────────

with open('golds-ufsc.json') as f:
    descriptor = json.load(f)

packet_types = {}
for link in descriptor['links']:
    for pkt_type in link['types']:
        id_field = next(f for f in pkt_type['fields'] if f['id'] == 'pkt_id')
        packet_types[id_field['value']] = pkt_type


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_callsign(cs: str) -> list:
    """Pack a callsign into exactly 7 bytes, space-padded."""
    cs = cs[:7].ljust(7)
    return [ord(c) for c in cs]

def u16(val: int) -> list:
    """Unsigned 16-bit big-endian."""
    val = val & 0xFFFF
    return [(val >> 8) & 0xFF, val & 0xFF]

def s16(val: int) -> list:
    """Signed 16-bit big-endian (two's complement)."""
    return u16(val & 0xFFFF)

def u32(val: int) -> list:
    """Unsigned 32-bit big-endian."""
    return [(val >> 24) & 0xFF,
            (val >> 16) & 0xFF,
            (val >>  8) & 0xFF,
             val        & 0xFF]

def get(decoded, field_id):
    """Shortcut: get a field's value from a decoded packet."""
    return decoded['fields'][field_id]['value']


# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────

class TestUnknownPacketID(unittest.TestCase):

    def test_unknown_id_returns_error(self):
        payload = bytes([0xFF] + [0] * 10)
        result = decode_packet(payload)
        self.assertIn('error', result)
        self.assertIn('255', result['error'])

    def test_unknown_id_3_returns_error(self):
        """ID 3 is not defined in the descriptor (gap between VHF types 2 and UHF type 16)."""
        payload = bytes([3] + [0] * 10)
        result = decode_packet(payload)
        self.assertIn('error', result)

    def test_empty_payload_does_not_crash(self):
        """A zero-byte payload has no pkt_id to read — should not raise, should error gracefully."""
        try:
            result = decode_packet(bytes([]))
        except (IndexError, KeyError):
            self.fail("decode_packet raised an exception on empty payload")

    def test_result_has_no_fields_on_error(self):
        payload = bytes([99] + [0] * 5)
        result = decode_packet(payload)
        self.assertNotIn('fields', result)


class TestPacketIDRecognition(unittest.TestCase):

    def test_all_valid_ids_are_recognized(self):
        """Every defined packet ID in the descriptor must decode without an 'error' key."""
        valid_ids = list(packet_types.keys())
        for pkt_id in valid_ids:
            # build a minimal payload of 110 zero bytes — long enough for any packet type
            payload = bytes([pkt_id] + [0] * 110)
            result = decode_packet(payload)
            self.assertNotIn('error', result,
                             msg=f"Packet ID {pkt_id} returned an error: {result}")

    def test_packet_type_name_matches_descriptor(self):
        payload = bytes([0] + [0] * 70)
        result = decode_packet(payload)
        self.assertEqual(result['packet_type'], 'EPS data')

        payload = bytes([16] + [0] * 110)
        result = decode_packet(payload)
        self.assertEqual(result['packet_type'], 'General telemetry')


class TestCallsignDecoding(unittest.TestCase):

    def test_callsign_decoded_correctly(self):
        cs = make_callsign('PY0EUF')
        payload = bytes([0] + cs + [0] * 60)
        result = decode_packet(payload)
        self.assertIn('PY0EUF', get(result, 'pkt_src_adr'))

    def test_short_callsign_padded(self):
        cs = make_callsign('PP5')
        payload = bytes([0] + cs + [0] * 60)
        result = decode_packet(payload)
        decoded_cs = get(result, 'pkt_src_adr')
        self.assertTrue(decoded_cs.startswith('PP5'))

    def test_callsign_exactly_7_bytes(self):
        cs = make_callsign('ABCDEFG')
        payload = bytes([0] + cs + [0] * 60)
        result = decode_packet(payload)
        # 7 chars total: the conversion slices pkt[1:8]
        self.assertEqual(len(get(result, 'pkt_src_adr')), 7)


class TestEPSData(unittest.TestCase):
    """Tests for packet type ID 0 — EPS data (VHF)."""

    def _make_eps(self,
                  timestamp=0, mcu_temp=0, mcu_curr=0, rst_count=0,
                  bat_volt=0, bat_curr=0, bat_charge=0):
        cs = make_callsign('PY0EUF')
        payload = (
            [0]             +   # pkt_id
            cs              +   # bytes 1-7
            u32(timestamp)  +   # bytes 8-11
            u16(mcu_temp)   +   # bytes 12-13
            u16(mcu_curr)   +   # bytes 14-15
            u16(rst_count)  +   # bytes 16-17
            u16(0) * 9      +   # sp_volt x3, sp_curr x6  (18 bytes, 38-35=18 but flattened)
            [0, 0, 0]       +   # mppt duty cycles (36-38)
            u16(0)          +   # mppt_sp_volt (39-40)
            u16(0)          +   # bus_volt (41-42)
            u16(0) * 7      +   # rtd temps (43-56)
            u16(bat_volt)   +   # bytes 57-58
            s16(bat_curr)   +   # bytes 59-60 (signed)
            u16(bat_charge) +   # bytes 61-62
            [0, 0]              # heater duty cycles (63-64)
        )
        return bytes(payload)

    def test_timestamp_zero(self):
        result = decode_packet(self._make_eps(timestamp=0))
        self.assertEqual(get(result, 'eps_timestamp'), 0)

    def test_timestamp_max_u32(self):
        result = decode_packet(self._make_eps(timestamp=0xFFFFFFFF))
        self.assertEqual(get(result, 'eps_timestamp'), 0xFFFFFFFF)

    def test_timestamp_realistic(self):
        result = decode_packet(self._make_eps(timestamp=1748638528))
        self.assertEqual(get(result, 'eps_timestamp'), 1748638528)

    def test_mcu_temp_typical(self):
        result = decode_packet(self._make_eps(mcu_temp=298))
        self.assertEqual(get(result, 'eps_mcu_temp'), 298)

    def test_mcu_temp_max_u16(self):
        result = decode_packet(self._make_eps(mcu_temp=0xFFFF))
        self.assertEqual(get(result, 'eps_mcu_temp'), 0xFFFF)

    def test_battery_voltage_typical(self):
        result = decode_packet(self._make_eps(bat_volt=8120))
        self.assertEqual(get(result, 'eps_bat_volt'), 8120)

    def test_battery_voltage_zero(self):
        result = decode_packet(self._make_eps(bat_volt=0))
        self.assertEqual(get(result, 'eps_bat_volt'), 0)

    def test_battery_current_positive_charging(self):
        """Positive current = battery is charging."""
        result = decode_packet(self._make_eps(bat_curr=250))
        self.assertEqual(int(get(result, 'eps_bat_curr')), 250)

    def test_battery_current_negative_discharging(self):
        """Negative current = battery is discharging. Stored as signed 16-bit."""
        result = decode_packet(self._make_eps(bat_curr=-300))
        self.assertEqual(int(get(result, 'eps_bat_curr')), -300)

    def test_battery_current_minus_one(self):
        result = decode_packet(self._make_eps(bat_curr=-1))
        self.assertEqual(int(get(result, 'eps_bat_curr')), -1)

    def test_battery_current_min_s16(self):
        result = decode_packet(self._make_eps(bat_curr=-32768))
        self.assertEqual(int(get(result, 'eps_bat_curr')), -32768)

    def test_reset_counter_increments(self):
        for n in [0, 1, 5, 255, 1000]:
            result = decode_packet(self._make_eps(rst_count=n))
            self.assertEqual(get(result, 'eps_mcu_rst_count'), n)

    def test_all_expected_fields_present(self):
        result = decode_packet(self._make_eps())
        expected = [
            'pkt_id', 'pkt_src_adr', 'eps_timestamp', 'eps_mcu_temp',
            'eps_mcu_curr', 'eps_mcu_rst_count', 'eps_sp_volt_mypx',
            'eps_bat_volt', 'eps_bat_curr', 'eps_bat_charge',
            'eps_bat_heater_1_dc', 'eps_bat_heater_2_dc'
        ]
        for field in expected:
            self.assertIn(field, result['fields'],
                          msg=f"Missing expected field: {field}")


class TestGeneralTelemetry(unittest.TestCase):
    """Tests for packet type ID 16 — General telemetry (UHF)."""

    def _make_telemetry(self, lat=0, lon=0, alt=0, bat_curr=0, timestamp=0):
        cs = make_callsign('PY0EUF')
        payload = (
            [16]             +   # pkt_id
            cs               +   # bytes 1-7
            u32(timestamp)   +   # bytes 8-11
            u16(300)         +   # obdh_mcu_temp 12-13
            u16(180)         +   # obdh_curr 14-15
            u16(5000)        +   # obdh_volt 16-17
            [0x01]           +   # obdh_mcu_last_rst_cause 18
            u16(5)           +   # obdh_mcu_rst_count 19-20
            [0x02]           +   # obdh_op_last_val_tc 21
            u16(298)         +   # ttc_radio1_temp 22-23
            u16(0)           +   # obdh_last_valid_tc_rssi 24-25
            u16(286)         +   # ant_temp 26-27
            u16(0x000F)      +   # ant_status 28-29
            u16(297)         +   # eps_mcu_temp 30-31
            u16(200)         +   # eps_mcu_curr 32-33
            [0x00]           +   # eps_mcu_last_rst_cause 34
            u16(2)           +   # eps_mcu_rst_count 35-36
            u16(0) * 9       +   # sp_volt x3, sp_curr x6 (37-54)
            [0, 0, 0]        +   # mppt dc 55-57
            u16(0)           +   # bus_volt 58-59
            u16(8120)        +   # bat_volt 60-61
            s16(bat_curr)    +   # bat_curr 62-63 (signed)
            s16(0)           +   # bat_avg_curr 64-65 (signed)
            u16(2000)        +   # bat_acc_curr 66-67
            u16(294)         +   # bat_mon_temp 68-69
            [0, 0]           +   # heater dc 70-71
            [1, 1, 0]        +   # main_edc, main_pl_state, sec_pl_state 72-74
            u32(0x1234)      +   # mem_data_log 75-78
            u32(0x80)        +   # mem_sec_sbcd 79-82
            u32(timestamp)   +   # sensor_read_ts 83-86
            u32(timestamp)   +   # pos_ts 87-90
            s16(lat)         +   # lat 91-92 (signed)
            s16(lon)         +   # lon 93-94 (signed)
            s16(alt)         +   # alt 95-96
            u32(0)           +   # last_tle_upd 97-100
            u32(0)               # last_contact 101-104
        )
        return bytes(payload)

    def test_latitude_positive(self):
        result = decode_packet(self._make_telemetry(lat=10))
        self.assertEqual(int(get(result, 'obdh_pos_lat')), 10)

    def test_latitude_negative_southern_hemisphere(self):
        result = decode_packet(self._make_telemetry(lat=-27))
        self.assertEqual(int(get(result, 'obdh_pos_lat')), -27)

    def test_latitude_boundary_plus_90(self):
        result = decode_packet(self._make_telemetry(lat=90))
        self.assertEqual(int(get(result, 'obdh_pos_lat')), 90)

    def test_latitude_boundary_minus_90(self):
        result = decode_packet(self._make_telemetry(lat=-90))
        self.assertEqual(int(get(result, 'obdh_pos_lat')), -90)

    def test_longitude_negative_western(self):
        result = decode_packet(self._make_telemetry(lon=-48))
        self.assertEqual(int(get(result, 'obdh_pos_lon')), -48)

    def test_longitude_positive_eastern(self):
        result = decode_packet(self._make_telemetry(lon=135))
        self.assertEqual(int(get(result, 'obdh_pos_lon')), 135)

    def test_altitude_typical_leo(self):
        """GOLDS-UFSC is in LEO, roughly 550km."""
        result = decode_packet(self._make_telemetry(alt=550))
        self.assertEqual(int(get(result, 'obdh_pos_alt')), 550)

    def test_altitude_zero(self):
        result = decode_packet(self._make_telemetry(alt=0))
        self.assertEqual(int(get(result, 'obdh_pos_alt')), 0)

    def test_battery_current_discharging(self):
        result = decode_packet(self._make_telemetry(bat_curr=-500))
        self.assertEqual(int(get(result, 'eps_bat_curr')), -500)

    def test_timestamp_propagates(self):
        result = decode_packet(self._make_telemetry(timestamp=1748638528))
        self.assertEqual(get(result, 'obdh_timestamp'), 1748638528)

    def test_packet_type_name(self):
        result = decode_packet(self._make_telemetry())
        self.assertEqual(result['packet_type'], 'General telemetry')


class TestPingAnswer(unittest.TestCase):
    """Tests for packet type ID 2 — Ping answer (VHF)."""

    def _make_ping(self, src='PY0EUF', requester='PY5XX  '):
        return bytes(
            [2]                       +
            make_callsign(src)        +
            make_callsign(requester)
        )

    def test_ping_recognized(self):
        result = decode_packet(self._make_ping())
        self.assertEqual(result['packet_type'], 'Ping answer')

    def test_source_callsign(self):
        result = decode_packet(self._make_ping(src='PY0EUF'))
        self.assertIn('PY0EUF', get(result, 'pkt_src_adr'))

    def test_requester_callsign(self):
        result = decode_packet(self._make_ping(requester='PY5XX'))
        self.assertIn('PY5XX', get(result, 'tc_req_adr'))

    def test_ping_has_exactly_3_fields(self):
        result = decode_packet(self._make_ping())
        self.assertEqual(len(result['fields']), 3)


class TestMessageBroadcast(unittest.TestCase):
    """Tests for packet type ID 1 — Message broadcast (VHF)."""

    def _make_message(self, msg: str) -> bytes:
        msg_bytes = [ord(c) for c in msg[:38].ljust(38)]
        return bytes(
            [1]                         +
            make_callsign('PY0EUF')     +   # src
            make_callsign('PY5XX  ')    +   # requester
            make_callsign('PY5YY  ')    +   # destination
            msg_bytes
        )

    def test_message_decoded(self):
        result = decode_packet(self._make_message('Hello from space!'))
        self.assertIn('Hello from space!', get(result, 'broadcast_msg'))

    def test_empty_message(self):
        result = decode_packet(self._make_message(''))
        self.assertIsInstance(get(result, 'broadcast_msg'), str)

    def test_message_type_name(self):
        result = decode_packet(self._make_message('test'))
        self.assertEqual(result['packet_type'], 'Message broadcast')


class TestTCFeedback(unittest.TestCase):
    """Tests for packet type ID 19 — TC feedback (UHF)."""

    def _make_tc_feedback(self, tc_pkt_id=0, timestamp=0):
        return bytes(
            [19]                     +
            make_callsign('PY0EUF') +
            make_callsign('PY5XX  ')+
            [tc_pkt_id]             +
            u32(timestamp)
        )

    def test_tc_feedback_recognized(self):
        result = decode_packet(self._make_tc_feedback())
        self.assertEqual(result['packet_type'], 'TC feedback')

    def test_tc_pkt_id_zero(self):
        result = decode_packet(self._make_tc_feedback(tc_pkt_id=0))
        self.assertEqual(get(result, 'tc_pkt_id'), 0)

    def test_tc_pkt_id_max(self):
        result = decode_packet(self._make_tc_feedback(tc_pkt_id=255))
        self.assertEqual(get(result, 'tc_pkt_id'), 255)

    def test_timestamp_decoded(self):
        result = decode_packet(self._make_tc_feedback(timestamp=1748638528))
        self.assertEqual(get(result, 'tc_feed_ts'), 1748638528)


class TestPayloadTruncation(unittest.TestCase):
    """What happens when the payload is too short for the expected fields."""

    def test_eps_data_too_short_produces_parse_error(self):
        """Only 5 bytes — fields beyond byte 4 will fail to parse."""
        payload = bytes([0] + make_callsign('PY0EUF') + [0] * 5)
        result = decode_packet(payload)
        # Some fields will be 'parse error: ...'
        parse_errors = [
            f for f in result['fields'].values()
            if isinstance(f['value'], str) and 'parse error' in f['value']
        ]
        self.assertGreater(len(parse_errors), 0)

    def test_partial_payload_still_returns_known_fields(self):
        """Fields that fit within the truncated payload should still decode."""
        payload = bytes([0] + make_callsign('PY0EUF') + [0x01, 0x2A] * 2)
        result = decode_packet(payload)
        # pkt_id and callsign are at bytes 0-7, should always be present
        self.assertEqual(get(result, 'pkt_id'), 0)
        self.assertIn('PY0EUF', get(result, 'pkt_src_adr'))


class TestByteOrderAndEndianness(unittest.TestCase):
    """Verify big-endian byte ordering is applied correctly throughout."""

    def test_u32_high_byte_first(self):
        """Timestamp 0x01020304 → bytes [0x01, 0x02, 0x03, 0x04]."""
        ts = 0x01020304
        payload = bytes([0] + make_callsign('PY0EUF') + u32(ts) + [0] * 60)
        result = decode_packet(payload)
        self.assertEqual(get(result, 'eps_timestamp'), 0x01020304)

    def test_u16_high_byte_first(self):
        """Battery voltage 0x1F40 (8000) → bytes [0x1F, 0x40]."""
        payload = bytes([0] + make_callsign('PY0EUF') + [0]*49 + [0x1F, 0x40] + [0]*8)
        result = decode_packet(payload)
        # eps_bat_volt is at initial_pos 57
        # 1 (id) + 7 (cs) + 4 (ts) + 2+2+2 (mcu) + 18 (sp) + 3+2+2 (mppt) + 14 (rtd) = 57
        self.assertEqual(get(result, 'eps_bat_volt'), 8000)

    def test_s16_negative_sign_correct(self):
        """Signed -1 is stored as 0xFF 0xFF in big-endian."""
        # Build a minimal EPS packet with bat_curr = -1 at bytes 59-60
        pre  = [0] + make_callsign('PY0EUF') + [0]*51   # bytes 0-58
        curr = [0xFF, 0xFF]                              # bytes 59-60 = -1
        post = [0]*4                                     # bytes 61-64
        result = decode_packet(bytes(pre + curr + post))
        self.assertEqual(int(get(result, 'eps_bat_curr')), -1)


if __name__ == '__main__':
    unittest.main(verbosity=2)