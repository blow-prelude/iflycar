#include "switch2.h"

// =========================================================================
// 主函数与状态机主循环
// =========================================================================
int main(int argc, char **argv)
{
    ros::init(argc, argv, "switch_node");
    ros::NodeHandle nh;

    OURSWITCH ucar;
    ros::AsyncSpinner spinner(1);
    spinner.start();

    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        switch (ucar.current_state)
        {
        case TEST_:
            break;
        case GOTOA_:
            ucar.GotoA();
            ucar.current_state = GOTOB_;
            break;
        case GOTOB_:
            ucar.GotoB();
            break;
        case XingHuoAI_:
            ucar.XingHuoAI();
            break;
        case GOTOC1_:
            ucar.GotoC(1);
            break;
        case GOTOC2_:
            ucar.GotoC(2);
            break;
        case Gazebo_:
            ucar.Gazebo();
            break;
        case GOTOD_:
            ucar.GotoD();
            break;
        case VISION_LINE_:
            ucar.vision_line();
            break;
        }
        loop_rate.sleep();
    }

    spinner.stop();
    return 0;
}

