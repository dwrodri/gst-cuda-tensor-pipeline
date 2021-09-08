import os, sys
import math, time
import itertools
import contextlib
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
import torch, torchvision

import ghetto_nvds

frame_format = "RBGA"
device = torch.device(0)
start_time, frames_processed = None, 0
image_batch, batch_size = [], 8

# context manager to help keep track of ranges of time, using NVTX
@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


def on_frame_probe(pad, info):
    global start_time, frames_processed
    start_time = start_time or time.time()

    global image_batch

    if not image_batch:
        torch.cuda.nvtx.range_push("batch")
        torch.cuda.nvtx.range_push("create_batch")

    buf = info.get_buffer()
    print(f"[{buf.pts / Gst.SECOND:6.2f}]")

    image_tensor = buffer_to_image_tensor(buf, pad.get_current_caps())
    image_batch.append(image_tensor)

    torch.cuda.nvtx.range_pop()  # batch
    return Gst.PadProbeReturn.OK


def buffer_to_image_tensor(buf, caps):
    with nvtx_range("buffer_to_image_tensor"):
        caps_structure = caps.get_structure(0)
        height, width = caps_structure.get_value("height"), caps_structure.get_value(
            "width"
        )

        is_mapped, map_info = buf.map(Gst.MapFlags.READ)
        if is_mapped:
            try:
                source_surface = ghetto_nvds.NvBufSurface(map_info)
                torch_surface = ghetto_nvds.NvBufSurface(map_info)

                dest_tensor = torch.zeros(
                    (
                        torch_surface.surfaceList[0].height,
                        torch_surface.surfaceList[0].width,
                        4,
                    ),
                    dtype=torch.uint8,
                    device=device,
                )

                torch_surface.struct_copy_from(source_surface)
                assert source_surface.numFilled == 1
                assert source_surface.surfaceList[0].colorFormat == 19  # RGBA

                # make torch_surface map to dest_tensor memory
                torch_surface.surfaceList[0].dataPtr = dest_tensor.data_ptr()

                # copy decoded GPU buffer (source_surface) into Pytorch tensor (torch_surface -> dest_tensor)
                torch_surface.mem_copy_from(source_surface)
            finally:
                buf.unmap(map_info)

            return dest_tensor[:, :, :3]


Gst.init("")
pipeline = Gst.parse_launch(
    f"""
    filesrc location={sys.argv[1]} num-buffers=256 !
    decodebin !
    nvvideoconvert !
    video/x-raw(memory:NVMM),format=RGBA !
    fakesink name=s
"""
)

pipeline.get_by_name("s").get_static_pad("sink").add_probe(
    Gst.PadProbeType.BUFFER, on_frame_probe
)

pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        msg = pipeline.get_bus().timed_pop_filtered(
            Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        if msg:
            text = msg.get_structure().to_string() if msg.get_structure() else ""
            msg_type = Gst.message_type_get_name(msg.type)
            print(f"{msg.src.name}: [{msg_type}] {text}")
            break
finally:
    finish_time = time.time()
    open(f"logs/{os.path.splitext(sys.argv[0])[0]}.pipeline.dot", "w").write(
        Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    )
    pipeline.set_state(Gst.State.NULL)

for tensor in image_batch:
    print(tensor)
