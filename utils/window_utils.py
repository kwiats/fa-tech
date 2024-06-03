from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID

def get_window_bounds(app_name):
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    for window in window_list:
        if app_name.lower() in window.get('kCGWindowOwnerName', '').lower():
            bounds = window.get('kCGWindowBounds', {})
            x, y, width, height = bounds.get('X', 0), bounds.get('Y', 0), bounds.get('Width', 0), bounds.get('Height', 0)
            return [x, y, width, height]
    return None

def list_windows():
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    windows = [(window.get('kCGWindowOwnerName', ''), window.get('kCGWindowName', '')) for window in window_list]
    return windows
