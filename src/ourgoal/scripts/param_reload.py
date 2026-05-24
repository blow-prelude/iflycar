#!/usr/bin/python3
import rospy
import rospkg
import dynamic_reconfigure.client
from ourgoal.srv import *
import yaml

class Config:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = {}
        self.load_parameters()

    def load_parameters(self):
        with open(self.file_path, 'r',encoding='utf-8') as file:
            try:
                self.config = yaml.safe_load(file)
                print(self.config)
            except yaml.YAMLError as exc:
                print("error")

    def get_param(self, param_name, default_value=None):
        return self.parameters.get(param_name, default_value)

def callback(config):
    rospy.loginfo("Config set done!!!")
def switch_file(value):
    rospack = rospkg.RosPack()
    package_path = rospack.get_path("ourgoal") + "/scripts/"
    switcher = {
        0: "0.yaml",
        1: "1.yaml",
        2: "2.yaml",
       
    }
    return package_path + switcher.get(value, "Value is unknown")

def doReq(req):
    # 解析提交的数据
    try:
        file_path = switch_file(req.ask)
        # print(file_path)
        c = Config(file_path)
        print(c.config)
        client.update_configuration(c.config)
        # client.update_configuration({"acc_lim_x" : 0.7, "acc_lim_theta" : 2.3, "max_vel_x" : 16.25,  "weight_acc_lim_theta": 60.0, "weight_max_vel_theta": 40.0, "weight_acc_lim_x" : 5})
        resp = True
    except Exception as e:
        print("doReq-error: ",str(e))

    return resp

if __name__ == "__main__":

    try:
        rospy.init_node("dynamic_client")
        print("waiting")
        rospy.wait_for_service("/move_base/TebLocalPlannerROS/set_parameters")
        print("waited")
        client = dynamic_reconfigure.client.Client("/move_base/TebLocalPlannerROS", timeout=10, config_callback=callback)
        server = rospy.Service("/param_reload", srv_reload ,doReq)
        print("power !")

        #timeout是超时时间
        #callback是使用一个回调函数，这个是可以可选的，在update之后可以进入回调操作（比如输出一些日志）
        rospy.spin()
    except Exception as e:
        print("param_main-error: ",str(e))
