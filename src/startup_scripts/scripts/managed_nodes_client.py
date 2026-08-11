#!/usr/bin/env python3

"""Provide one SetBool endpoint for enabling or disabling managed nodes."""

from __future__ import annotations

import threading

import rospy
from std_srvs.srv import SetBool, SetBoolResponse


class ManagedNodesClient:
    def __init__(self) -> None:
        self._timeout = float(rospy.get_param("~service_wait_timeout", 2.0))
        if self._timeout <= 0.0:
            raise ValueError("~service_wait_timeout must be > 0")
        self._call_lock = threading.Lock()

        self._services = {
            "traffic_light": rospy.get_param(
                "~traffic_light_service", "/traffic_light/set_enabled"
            ),
            "image_process": rospy.get_param(
                "~image_process_service", "/image_process/set_enabled"
            ),
            "vision_line_node": rospy.get_param(
                "~vision_line_service", "/vision_line_node/set_enabled"
            ),
        }
        self._proxies = {
            name: rospy.ServiceProxy(service_name, SetBool)
            for name, service_name in self._services.items()
        }
        self._service = rospy.Service(
            "~set_enabled", SetBool, self._set_enabled_callback
        )
        self._traffic_light_enable_param = rospy.get_param(
            "~traffic_light_enable_param", "/start_traffic_light_det"
        )
        poll_rate = float(rospy.get_param("~parameter_poll_rate", 10.0))
        if poll_rate <= 0.0:
            raise ValueError("~parameter_poll_rate must be > 0")
        self._last_traffic_light_enabled = None
        self._parameter_timer = rospy.Timer(
            rospy.Duration(1.0 / poll_rate), self._parameter_timer_callback
        )
        rospy.loginfo("Managed-node control service is ready")

    def _read_bool_parameter(self, name: str) -> bool | None:
        value = rospy.get_param(name, 0)
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return value == 1
        rospy.logwarn_throttle(
            5.0, "%s must be bool or integer 0/1; got %r", name, value
        )
        return None

    def _parameter_timer_callback(self, _event: rospy.timer.TimerEvent) -> None:
        enabled = self._read_bool_parameter(self._traffic_light_enable_param)
        if enabled is None or enabled == self._last_traffic_light_enabled:
            return

        with self._call_lock:
            succeeded, message = self._call_target("traffic_light", enabled)
        if succeeded:
            self._last_traffic_light_enabled = enabled
            rospy.loginfo(
                "%s=%s -> traffic_light %s",
                self._traffic_light_enable_param,
                int(enabled),
                message,
            )

    def _call_target(self, name: str, enabled: bool) -> tuple[bool, str]:
        service_name = self._services[name]
        try:
            rospy.wait_for_service(service_name, timeout=self._timeout)
            response = self._proxies[name](data=enabled)
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Failed to call %s: %s", service_name, error)
            return False, f"error({error})"

        if not response.success:
            return False, f"rejected({response.message})"
        return True, response.message or "ok"

    def _set_enabled_callback(self, request: SetBool.Request) -> SetBoolResponse:
        if request.data:
            order = ("traffic_light", "image_process", "vision_line_node")
        else:
            order = ("vision_line_node", "image_process", "traffic_light")

        all_succeeded = True
        details = []
        with self._call_lock:
            for name in order:
                succeeded, message = self._call_target(name, request.data)
                all_succeeded = all_succeeded and succeeded
                details.append(f"{name}={'ok' if succeeded else message}")

        summary = "; ".join(details)
        rospy.loginfo("Managed nodes enabled=%s: %s", request.data, summary)
        if all_succeeded:
            self._last_traffic_light_enabled = self._read_bool_parameter(
                self._traffic_light_enable_param
            )
        return SetBoolResponse(success=all_succeeded, message=summary)


def main() -> None:
    rospy.init_node("managed_nodes")
    ManagedNodesClient()
    rospy.spin()


if __name__ == "__main__":
    main()
