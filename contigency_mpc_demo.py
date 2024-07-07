import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

def masspoint_step(x, u):
    x = x + u 
    return x

def obstacle_step(x, u):
    x = x + 0.25 * u
    return x

def obstalce_obs_max(y, i):
    obstacle_obs_max= y + (10 - i) * 0.25 
    return min(1, obstacle_obs_max)

def rollout(x0, u_seq):
    masspoint_traj = [x0]
    for u in u_seq:
        masspoint_traj += [masspoint_step(masspoint_traj[-1], u)]
    
    return masspoint_traj

def RMPC_controller(y, y_obstacle_max, N):

    u = [max(0,y_obstacle_max) / 10] * N
    y = rollout(y, u)
    return u, y

def CMPC_controller(y, y_obstacle_max, N, Pc = 0.25):
    u0 = max(0,y_obstacle_max) * Pc / (Pc + 10 +1e-6)
    un = [u0] + [0] * (N-1)
    uc = [u0] + [u0 / Pc] * (N-1)

    yc = rollout(y, uc)
    yn = rollout(y, un)
    return uc, yc

def main():
    n_steps = 10
    u_masspoint_history = [0]
    y_masspoint_history = [0]
    y_obstacle_history = [-1]
    u_obstacle_history = [0] * n_steps
    y_obstacle_max_history = [obstalce_obs_max(y_obstacle_history[-1], 0)]
    
    fig = plt.figure(1, figsize=[8,6])
    ax_state = fig.add_subplot(211, aspect='equal')
    ax_ctrl = fig.add_subplot(212)
    
    for k in range(n_steps):
        ax_state.cla()
        ax_state.set_title("Robust MPC")
        ax_state.set_ylabel("y")
        ax_state.grid()
        ax_ctrl.cla()
        ax_ctrl.set_xlabel("t = " + str(k))
        ax_ctrl.set_ylabel("u")
        ax_ctrl.grid()

        u_masspoint_predict, y_masspoint_predict = RMPC_controller(y_masspoint_history[-1], y_obstacle_max_history[-1], n_steps - k)
        plot_trajectory(y_masspoint_predict, u_masspoint_predict, k+1, style = "bs")
        
        u_masspoint_history += [u_masspoint_predict[0]]
        y_masspoint_history += [masspoint_step(y_masspoint_history[-1], u_masspoint_history[-1])]
        plot_trajectory(y_masspoint_history, u_masspoint_history, 0, style = "g*" )
 
        y_obstacle_history += [ obstacle_step(y_obstacle_history[-1], u_obstacle_history[k]) ]
        y_obstacle_max_history += [ obstalce_obs_max(y_obstacle_history[-1], k)]
        plot_obstacle(y_obstacle_max_history[-1], y_obstacle_history[-1])

        ax_state.legend(["predict_y", "history_t", "obstacle"])
        ax_ctrl.legend(["predict_ctrl", "history_ctrl"])

        fig.savefig("imgs/R" + str(k))
        plt.pause(1)

    plt.show()

def plot_trajectory(y_masspoint,u_masspoint, start_step = 0, style = "bs"):
    fig = plt.gcf()
    ax_state, ax_ctrl = fig.get_axes()

    ax_state.set_xlim([0, 11])
    ax_state.set_ylim([-1.8, 1.8])
    ax_state.plot(range(start_step, start_step+len(y_masspoint)), y_masspoint, style)

    ax_ctrl.set_xlim([0, 11])
    ax_ctrl.set_ylim([-0.15, 0.15])
    ax_ctrl.plot(range(start_step, start_step+len(u_masspoint)), u_masspoint, style)
    
def plot_obstacle(y_obstacle_max, y_obstacle):
    fig = plt.gcf()
    ax_state, ax_ctrl = fig.get_axes()
    y_posible_length = max(y_obstacle_max - y_obstacle , 0.01)
    obstacle_posible_rect= patches.Rectangle((10,y_obstacle-0.05), 0.5, y_posible_length, color = "salmon")
    obstacle_real_rect= patches.Rectangle((10,y_obstacle-0.05), 0.5, 0.05, color = "red")
    ax_state.add_patch(obstacle_posible_rect)
    ax_state.add_patch(obstacle_real_rect)


def save_animation():

    save_name = 'demo.mp4'
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    fps = 1
    width = 1620
    height = 600 
    out = cv2.VideoWriter(save_name, fourcc, fps, (width, height))
    
    b_img = np.zeros((600,20,3),dtype=np.uint8)

    for i in range(10):
        c_file_name = "./imgs/C" + str(i) + ".png"
        c_img = cv2.imread(c_file_name)
        r_file_name = "./imgs/R" + str(i) + ".png"
        r_img = cv2.imread(r_file_name)
        img = np.concatenate([c_img, b_img, r_img],axis=1)
        out.write(img)
    out.release()

    print("save finished")


if __name__ == "__main__":
    # main()
    save_animation()
    