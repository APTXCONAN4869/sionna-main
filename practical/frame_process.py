import numpy as np
import torch
from math import ceil

class BinaryFramePacker:
    """
    output: List of torch.Tensor
        frame tensor shape = [8, 1, 1, 624]
    """

    def __init__(self,
                 file_path,
                 bits_per_slot=624,
                 slots_per_frame=8,
                 header_bits_cfg=(16, 16, 16, 8)):
        """
        header_bits_cfg:
            (frame_id_bits, total_frames_bits, valid_bits_bits, reserved_bits)
        """
        self.file_path = file_path
        self.bits_per_slot = bits_per_slot
        self.slots_per_frame = slots_per_frame
        self.data_slots = slots_per_frame - 1
        self.bits_per_frame = self.data_slots * bits_per_slot
        self.header_bits_cfg = header_bits_cfg

        self._load_file()


    @staticmethod
    def _int_to_bits(x, width):
        return np.array(list(np.binary_repr(x, width=width)), dtype=np.uint8)

    def _load_file(self):
        with open(self.file_path, "rb") as f:
            raw_bytes = f.read()

        self.file_bits = np.unpackbits(
            np.frombuffer(raw_bytes, dtype=np.uint8)
        )
        self.total_bits = len(self.file_bits)


    def pack(self):
        frames = []

        ptr = 0
        frame_id = 0
        total_frames = ceil(self.total_bits / self.bits_per_frame)

        while ptr < self.total_bits:
            remaining = self.total_bits - ptr
            valid_bits = min(self.bits_per_frame, remaining)

            # -------------------------
            # payload
            # -------------------------
            payload = self.file_bits[ptr: ptr + valid_bits]
            ptr += valid_bits

            payload_padded = np.pad(
                payload,
                (0, self.bits_per_frame - len(payload)),
                mode="constant"
            )

            # -------------------------
            # slot 0: header
            # -------------------------
            fid_bits, tf_bits, vb_bits, rsv_bits = self.header_bits_cfg

            header_bits = np.concatenate([
                self._int_to_bits(frame_id, fid_bits),
                self._int_to_bits(total_frames, tf_bits),
                self._int_to_bits(valid_bits, vb_bits),
                self._int_to_bits(0, rsv_bits),
            ])

            if len(header_bits) > self.bits_per_slot:
                raise ValueError("Header bits exceed slot size")

            slot0 = np.pad(
                header_bits,
                (0, self.bits_per_slot - len(header_bits)),
                mode="constant"
            )

            # -------------------------
            # slot 1~N: payload
            # -------------------------
            slots_data = payload_padded.reshape(
                self.data_slots, self.bits_per_slot
            )

            # -------------------------
            # assemble frame
            # -------------------------
            frame_bits = np.vstack([
                slot0.reshape(1, self.bits_per_slot),
                slots_data
            ])  # (8, 624)

            frame_tensor = torch.from_numpy(frame_bits).float()
            frame_tensor = frame_tensor.unsqueeze(1).unsqueeze(2)
            # -> [8, 1, 1, 624]

            frames.append(frame_tensor)
            frame_id += 1

        return frames

class BinaryFrameUnpacker:
    """
    - input: rx_bits [N*8, 1, 1, 624]
    - automatically parse slot0
    - deduplicate, count missing frames
    - recover original file when complete
    """

    def __init__(self,
                 bits_per_slot=624,
                 slots_per_frame=8,
                 header_bits_cfg=(16, 16, 16, 8)):
        self.bits_per_slot = bits_per_slot
        self.slots_per_frame = slots_per_frame
        self.data_slots = slots_per_frame - 1
        self.bits_per_frame = self.data_slots * bits_per_slot
        self.header_bits_cfg = header_bits_cfg

        self.reset()

    def reset(self):
        self.buffer = {}          # frame_id -> payload_bits
        self.total_frames = None  # 从帧头解析得到
        self.valid_bits_map = {}  # frame_id -> valid_bits

    @staticmethod
    def _bits_to_int(bits):
        return int("".join(bits.astype(str)), 2)

    # ------------------------------------------------
    # analyse slot0
    # ------------------------------------------------
    def _parse_header(self, slot0_bits):
        fid_w, tf_w, vb_w, rsv_w = self.header_bits_cfg

        idx = 0
        frame_id = self._bits_to_int(slot0_bits[idx:idx+fid_w])
        idx += fid_w

        total_frames = self._bits_to_int(slot0_bits[idx:idx+tf_w])
        idx += tf_w

        valid_bits = self._bits_to_int(slot0_bits[idx:idx+vb_w])

        return frame_id, total_frames, valid_bits

    def _header_sanity_check(self, frame_id, total_frames, valid_bits):
        MAX_FRAMES = 8192

        # frame_id
        if frame_id < 0 or frame_id >= MAX_FRAMES:
            return False, 1

        # total_frames
        if total_frames <= 0 or total_frames > MAX_FRAMES:
            return False, 2

        # valid_bits
        if valid_bits <= 0 or valid_bits > self.bits_per_frame:
            return False, 3

        # frame_id 必须小于 total_frames
        if frame_id >= total_frames:
            return False, 4
        # total_frames 一致性
        if self.total_frames is not None and total_frames != self.total_frames:
            return False, 5

        # valid_bits 形态约束
        if frame_id != total_frames - 1:
            if valid_bits != self.bits_per_frame:
                return False, 6

        return True, 0

    def push(self, rx_bits):
        """
        rx_bits: torch.Tensor [N*8, 1, 1, 624]
        """
        if rx_bits.device.type != 'cpu':
            rx_bits = rx_bits.detach().cpu()
        rx_bits = rx_bits.numpy().astype(np.uint8)
        num_slots = rx_bits.shape[0]
        assert num_slots % self.slots_per_frame == 0, "Slots number not aligned with slots_per_frame"

        num_frames = num_slots // self.slots_per_frame

        rx_bits = rx_bits.reshape(
            num_frames,
            self.slots_per_frame,
            self.bits_per_slot
        )  # [F, 8, 624]

        new_frames = 0
        dropped_frames = 0
        for i in range(num_frames):
            frame = rx_bits[i]

            # -------- slot0 --------
            slot0 = frame[0]
            frame_id, total_frames, valid_bits = self._parse_header(slot0)
            is_valid, error_code = self._header_sanity_check(frame_id, total_frames, valid_bits)
            if not is_valid:
                dropped_frames += 1
                if error_code == 1:
                    print(f"[WARN] frame_id not valid: {frame_id} | {total_frames} | {valid_bits}")
                    continue
                elif error_code == 2:
                    print(f"[WARN] total_frames not valid: {frame_id} | {total_frames} | {valid_bits}")
                    continue
                elif error_code == 3:
                    print(f"[WARN] valid_bits not valid: {frame_id} | {total_frames} | {valid_bits}")
                    continue
                elif error_code == 4:
                    print(f"[WARN] frame_id {frame_id} >= total_frames {total_frames}")
                    continue
                elif error_code == 5:
                    print(f"[WARN] Inconsistent total_frames {total_frames}")
                    continue
                elif error_code == 6:
                    print(f"[WARN] Invalid valid_bits {valid_bits} for frame_id {frame_id}")
                    continue
            if self.total_frames is None:
                self.total_frames = total_frames

            # -------- remove duplicates --------
            if frame_id in self.buffer:
                print(f"[WARN] Drop duplicate frame {frame_id}")
                continue

            # -------- payload --------
            payload_bits = frame[1:].reshape(-1)  # 7*624

            self.buffer[frame_id] = payload_bits
            self.valid_bits_map[frame_id] = valid_bits
            new_frames += 1
            if new_frames == 1:
                print(f"[INFO] Received first new frame:")
                print(f"       frame_id: {frame_id}")
                print(f"       total_frames: {total_frames}")
                print(f"       valid_bits: {valid_bits}")

        return new_frames


    def is_complete(self):
        if self.total_frames is None:
            return False
        return len(self.buffer) == self.total_frames

    def missing_frames(self):
        if self.total_frames is None:
            return None
        return self.total_frames - len(self.buffer)

    def recover_file(self, output_path="recv.bmp"):
        if not self.is_complete():
            raise RuntimeError("Frames not complete yet")

        bits = []

        for fid in sorted(self.buffer.keys()):
            payload = self.buffer[fid]
            valid = self.valid_bits_map[fid]
            bits.append(payload[:valid])

        all_bits = np.concatenate(bits)
        all_bytes = np.packbits(all_bits)

        with open(output_path, "wb") as f:
            f.write(all_bytes)

        return output_path



# packer = BinaryFramePacker("d:/sionna-main/practical/file.bmp")
# frames = packer.pack()
# print('Total frames:', len(frames))          # 帧数
# print('Frame shape:', frames[0].shape)      # torch.Size([8, 1, 1, 624])
# b = torch.cat(frames, dim=0)  # [num_frames,1,1,624]
# slots_per_frame = 8
# batch_frames = []
# ptr = 0
# while ptr<b.shape[0]:   
#     batch_frames.append(b[ptr:ptr+slots_per_frame])
#     ptr += slots_per_frame

# unpacker = BinaryFrameUnpacker()
# frame_id = 0
# while True:
#     if frame_id >= len(batch_frames):
#         print("No more frames to receive.")
#         break
#     rx_bits = batch_frames[frame_id % len(batch_frames)]  # 模拟每次接收 1 帧
#     new = unpacker.push(rx_bits)   # rx_bits.shape == [8*n,1,1,624]
#     # print(f"Number of New frames received: {new}")
#     if unpacker.is_complete():
#         unpacker.recover_file("d:/sionna-main/practical/recv.bmp")
#         print("File recovered!")
#         break
#     else:
#         print("Missing frames:", unpacker.missing_frames())
#     frame_id += 1