#include "action_managers/staubliControlActionManager.h"
#include "staubli_tx60/SetTrajectoryParams.h"
#define ERROR_EPSILON 1E-2

template <class ActionSpec>
void StaubliControlActionManager <ActionSpec>::abortHard()
{
    //unable to query robot state -- stop robot if possible, this is an important error
    ROS_ERROR("%s ::Communications Failure! Error when determining end of movement. *****  All goals cancelled and robot queue reset.  Staubli Server shutdown!!!",actionName_.c_str());

    as_.setAborted(mResult.result,"Communications failure - could not query joint position\n");
    //communication failures are bad.
    //Shut down the action server if they occur.  Don't accept new goals.
    as_.shutdown();
    //Cancel existing motion of robot if possible
    staubli.ResetMotion();
    //kill entire server
    ros::shutdown();
}

template <class ActionSpec>
void StaubliControlActionManager<ActionSpec>::cancelAction()
{
    if(as_.isActive())
    {
        ROS_WARN("%s::cancelAction::Last action cancelled",actionName_.c_str());
        staubli.ResetMotion();
        as_.setPreempted(mResult.result,"Action preempted by another action");
        if(!actionName_.compare("setFollowTrajectory"))
        {
            ROS_WARN("%s::cancelAction::Cancelled follow trajectory",actionName_.c_str());
        }
    }
}

template <class ActionSpec>
bool StaubliControlActionManager<ActionSpec>::pollRobot(const std::vector<double> &goal_joints, StaubliState & state)
{
    if(staubli.IsWorking())
    {
        updateFeedback(state);
        as_.publishFeedback(mFeedback.feedback);

        //if(isRobotFinishedMoving())
        //{
            updateResult(state);

            if(hasReachedGoal(state))
            {
                as_.setSucceeded(mResult.result);
                ROS_INFO("%s GOAL Reached", actionName_.c_str());
                return false;
            }
            else
            {
                if (staubli.IsJointQueueEmpty())
                {

                    staubli_tx60::StaubliMovementParams end_traj_params;
                    end_traj_params.distBlendPrev = 0;
                    end_traj_params.distBlendNext = 0;
                    end_traj_params.movementType = 0;
                    end_traj_params.jointVelocity = 0.4;
                    end_traj_params.jointAcc = 0.04;
                    end_traj_params.jointDec = 0.04;
                    end_traj_params.endEffectorMaxRotationalVel = 9999.0;
                    end_traj_params.endEffectorMaxTranslationVel = 9999.0;

                    bool goalOk = staubli.MoveJoints(this->mGoalValues,
                                            end_traj_params.movementType,
                                            end_traj_params.jointVelocity,
                                            end_traj_params.jointAcc = 0.04,
                                            end_traj_params.jointDec = 0.04,
                                            end_traj_params.endEffectorMaxTranslationVel,
                                            end_traj_params.endEffectorMaxRotationalVel,
                                            end_traj_params.distBlendPrev,
                                            end_traj_params.distBlendNext = 0
                                            );
                    if(!goalOk)
                    {
                        as_.setAborted(mResult.result, "Could not accept goal\n");
                        ROS_INFO("staubli::Goal rejected");
                        return false;
                    }

                }
            }
            return true;
//            else
//            {
//                as_.setAborted(mResult.result);
//                ROS_WARN("%s :: Staubli queue emptied prematurely, but goal was not reached\n",actionName_.c_str());
//            }
//            return false;
        //}
//        else
//        {
//            return true;
//        }
    }
    else
    {
        abortHard();
        return false;
    }
    return true;
}

template <class ActionSpec>
void StaubliControlActionManager<ActionSpec>::newGoalCallback()
{
    ROS_INFO("%s::newGoalCallback::New action sent",actionName_.c_str());
    const typename ActionSpecServer::GoalConstPtr goal = as_.acceptNewGoal();
    newGoalCallback(goal);
}

template <class ActionSpec>
void StaubliControlActionManager<ActionSpec>::newGoalCallback(const typename ActionSpecServer::GoalConstPtr  &goal)
{
    ROS_WARN("Staubli::%s: New goal recieved", actionName_.c_str());
    // If there wa sa previous goal of this type, preempt it
    //if(as_.isActive()){
    //    as_.setPreempted(mResult.result,"Received new goal");
    //}
    staubli.ResetMotion();

    mGoal.goal = *goal;
    if(!acceptGoal())
    {
      as_.setAborted(mResult.result,"Failed to start goal");
	ROS_ERROR("Staubli::%s: New goal failed", actionName_.c_str());
	
    }
    else
      {
	ROS_WARN("Staubli::%s: New goal accepted", actionName_.c_str());
      }
    // Preempt any other goals and mark this one as running
    activateAction();
}

template <class ActionSpec>
void StaubliControlActionManager<ActionSpec>::publishFeedback(StaubliState & state)
{
    if(running)
    {
      if(as_.isActive())
        {
            if(as_.isPreemptRequested())
            {
                ROS_INFO("Staubli::%s::Action preempted", actionName_.c_str());
                cancelAction();
            }
            running = pollRobot(mGoalValues, state);
        }
    }
    else
    {
      
        if(as_.isActive())
        {
            ROS_ERROR("%s is active, but not set to running!", actionName_.c_str());
        }
    }
}
