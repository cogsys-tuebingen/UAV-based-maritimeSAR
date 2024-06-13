import gi

import config

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject, GLib

import global_vars

STREAM_HEIGHT = 720
STREAM_WIDTH = 1280

FPS = 30
FRAME_DURATION = 1 / FPS * Gst.SECOND  # duration of a frame in nanoseconds


class RtspMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, coordinator):
        GstRtspServer.RTSPMediaFactory.__init__(self)

        self.coordinator = coordinator

        self.pipeline = ' ! '.join([
            f'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME caps=video/x-raw,format=BGR,width={STREAM_WIDTH},height={STREAM_HEIGHT}',
            'videoconvert ! video/x-raw,format=I420',
            'x264enc',
            'rtph264pay config-interval=1 name=pay0 pt=96',
        ])
        self.last_frame = None

    def do_create_element(self, url):
        print("# RTSP pipeline created: " + self.pipeline)
        return Gst.parse_launch(self.pipeline)

    def on_need_data(self, src, length):
        out_frame, timestamp = self.coordinator.get_next_frame()
        data = None
        if out_frame is not None:
            data = out_frame.tostring()
            self.last_frame = out_frame.copy()
        elif self.last_frame is not None:
            data = self.last_frame.tostring()

        if data is None:
            buf = Gst.Buffer.new_allocate(None, STREAM_HEIGHT*STREAM_WIDTH*3, None)
        else:
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)

        self.coordinator.inc_frame_number()

        buf.duration = FRAME_DURATION

        buf.pts = int(timestamp)
        buf.dts = int(timestamp)
        buf.offset = timestamp

        push_buffer_response = src.emit('push-buffer', buf)

        if push_buffer_response != Gst.FlowReturn.OK:
            print(push_buffer_response)

    def do_configure(self, rtsp_media):
        self.coordinator.reset_frame_number()
        # GstAppSrc — Easy way for applications to inject buffers into a pipeline
        #
        # see: https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gst-plugins-base-libs/html/gst-plugins-base-libs-appsrc.html
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class GstreamerRtspServer(GstRtspServer.RTSPServer):
    def __init__(self, coordinator, ip_address):
        super(GstreamerRtspServer, self).__init__()

        self.set_address(ip_address)
        self.factory = RtspMediaFactory(coordinator)
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/stream", self.factory)
        self.attach(None)
        print(f"# RTSP Server [{self.get_address()}] started and waiting for connections..")


def rtsp_stream_loop(coordinator):
    print("# Start RTSP server")

    Gst.init(None)
    s = GstreamerRtspServer(coordinator, config.LOCAL_IP)
    global_vars.loop = GLib.MainLoop() # moved to global_vars since KeyBoardInterrupt catched in ArmServer
    try:
        global_vars.loop.run()
    except KeyboardInterrupt:
        global_vars.loop.quit()
        global_vars.cancel_signal()

    print("# RTSP server stopped.")
