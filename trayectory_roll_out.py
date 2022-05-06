import numpy as np
import cv2
from matplotlib import pyplot as plt
import json



# Definition of model of robot
class modelo_turtle():
    def __init__(self):
        self.ChassisRadius = 0.220 -0.05        #[m]
        self.TrackWidth    = 0.287         #[m]  
        self.WheelRadius   = 0.033         #[m]
        self.WheelWidth    = 0.021         #[m]
        self.LinVelLims    = [-0.26,0.26]  #[m/s]
        #self.LinVelLims    = [-1,1]  #[m/s]
        self.AngVelLims    = [-1.82,1.82]  #[rad/s]
        self.Weight        = 1800          #[g]

# Definition of features of the map
class map_features():
    def __init__(self):
        self.heigth = 20        #[m]
        self.weigth = 20        #[m]
        self.pos_x = 10         #[m]
        self.pos_y = 10         #[m]
        self.resolution = 20    #[pix/m]


# Class for doing trayectory rollout
class TrajRollout:
    
    # Definition of features to solve the trayectory rollout
    def __init__(self,DynamicWindowNumVxSamples, DynamicWindowNumWSamples,DynamicWindowVxLims, DynamicWindowVthetaLims, DynamicWindowAxLims,
                 DynamicWindowAthetaLims, DynamicWindowTime, RolloutSampleTime, max_lookahead_dist, mapa):

                self.DynamicWindowNumVxSamples= DynamicWindowNumVxSamples
                self.DynamicWindowNumWSamples = DynamicWindowNumWSamples
                self.DynamicWindowVxLims = DynamicWindowVxLims
                self.DynamicWindowVthetaLims = DynamicWindowVthetaLims
                self.DynamicWindowAxLims = DynamicWindowAxLims
                self.DynamicWindowAthetaLims = DynamicWindowAthetaLims
                self.RolloutSampleTime = RolloutSampleTime
                self.DynamicWindowTime = DynamicWindowTime
                self.mapa= mapa
                self.path = []
                self.max_lookahead_dist = max_lookahead_dist
                
    
    #Funtion to choose the local among the whole path
    # The funtion will recieve the path
    def calc_local_path(self,r_xi):
        # r_xi: actual pose of robot in [x,y,tetha]
        
        vec_dist = (np.sum((self.path-r_xi[:2])**2,axis=1))**(1/2)
        
        
        pos_actual = vec_dist.argmin()
        
        
        vec_avaliable = vec_dist[pos_actual:-1] > self.max_lookahead_dist

        
        pos_final = np.where(vec_avaliable)
    
        
        if(pos_final[0].shape[0]==0):
            self.local_path = self.path
        else:
            self.local_path = self.path[pos_actual:pos_final[0][0]+pos_actual]
        
                
        
        
        
    # Funcion to calculate the dynamic window
    # The funcion will recieve the actual velocity vector
    def calcDynamicWindow(self, rdxi):
        # rdxi = twist robot  [x,y, theta]
        # [vx , 0, w]

        delta_t = self.DynamicWindowTime

        V_max_x = rdxi[0] + self.DynamicWindowAxLims[1] * delta_t
        V_min_x = rdxi[0] + self.DynamicWindowAxLims[0] * delta_t

        W_max = rdxi[2] + self.DynamicWindowAthetaLims[1] * delta_t
        W_min = rdxi[2] + self.DynamicWindowAthetaLims[0] * delta_t


        V_max_x = np.min([V_max_x,self.DynamicWindowVxLims[1]])
        V_min_x = np.max([V_min_x,self.DynamicWindowVxLims[0]])

        W_max = np.min([W_max,self.DynamicWindowVthetaLims[1]])
        W_min = np.max([W_min,self.DynamicWindowVthetaLims[0]])


        Vx = np.linspace(V_min_x,V_max_x,self.DynamicWindowNumVxSamples)
        W= np.linspace(W_min,W_max, self.DynamicWindowNumWSamples)

        [Vx,W] = np.meshgrid(Vx,W)

        r_dxi_cmds = np.vstack([Vx.flatten(),np.zeros_like(Vx.flatten()),W.flatten()])

        return  r_dxi_cmds
    
    
    # Funtion to load the path to the class, it will recieve the name of a
    # text file in which the path is
    def load_path(self, name_path):
        f = open (name_path,'r')
        mensaje = f.read()
        mensaje = mensaje.split('\n')
        mensaje.pop(-1)
        for idx, men in enumerate(mensaje):
            mensaje[idx] = (men.split('  '))
            mensaje[idx][0]= float(mensaje[idx][0])
            mensaje[idx][1]= float(mensaje[idx][1])
        self.path = np.array(mensaje,np.float_)

    # Funcion to interpolation to the path
    def path_interpolation(self,num):
        
        x2 = np.linspace(0, self.path.shape[0], num=self.path.shape[0]*num)
        x = np.arange(0,self.path.shape[0])
        path_x = np.interp(x2 , x, self.path[:,0]).reshape(x2.shape[0],1)
        path_y = np.interp(x2 , x, self.path[:,1]).reshape(x2.shape[0],1)
        path = np.concatenate((path_x,path_y),axis = 1)
        self.path = path

    # This funcion will calculeta the velocity of the robot for the actual time
    def calc_step(self, r_dxi, r_xi):
        # w_xi: twist del robot vx,vy,w
        # r_xi: pose actual del robot x,y,theta
        
        w_dxi = DiffDrive.convTwist2dBaseToWorld(r_dxi, r_xi[2])
        r_dxi_cmds = self.calcDynamicWindow(w_dxi)
        r_dxi_cmds = np.tile(r_dxi_cmds,(int(self.DynamicWindowTime/self.RolloutSampleTime),1,1))
        
        self.trayectorias = DiffDrive.calc_trayectory(r_xi,r_dxi_cmds,self.DynamicWindowTime,self.RolloutSampleTime)
        
        self.calc_local_path(r_xi)
        
        local = self.local_path

        arg_minimo = self.calc_cost(r_xi)

        if(arg_minimo != -1):
            r_dxi_cmds_final = r_dxi_cmds[0,:,arg_minimo]
        else:
            r_dxi_cmds_final = np.array([0,0,0])
        
        return r_dxi_cmds_final
        
        
    # This function calculate the funcion of cost of the trayectory roll out
    def calc_cost(self,r_xi):
        
        if(self.local_path.shape[0]!= 0):

            w_target = self.local_path[-1,:]
            
            cost_dist = self.calc_euclidian_dist_with_target(w_target)
            
            costo_obs = self.calc_cost_obstacle()
            
            minimo = (cost_dist+costo_obs).argmin()
            
            self.costos = cost_dist+costo_obs
            
            if((costo_obs== np.inf).all()):
                return -1


            return minimo
        else:
            return -1
        
    # This function calculate de cost depending in the obstacles
    def calc_cost_obstacle(self):
        trayectorias = self.trayectorias[:,:2,:]
        costo = np.zeros(trayectorias.shape[2])
        for i in range(trayectorias.shape[2]):
            costo[i]= self.mapa.checkcupancy(trayectorias[10:,:,i])
        return costo
        
    def calc_euclidian_dist_with_target(self,r_xi):
        # This function calculate the distance between the last positicion
        # of each trayectory and the last point of 

        last_point_trayectory = self.trayectorias[-1,:2,:]
        distance = np.sum((last_point_trayectory-r_xi[:2].reshape(2,1))**2,axis = 0)

        return distance
        
    
    
class DiffDrive():
    def __init__(self,wheelRadius,trackWidth):
        # Wheel radius in meters [m]
        # Distance from wheel to wheel in meters [m]
        self.wheelRadius = wheelRadius
        self.trackWidth = trackWidth
        
    
    
    def calcForwardKinematics(self, wr, wl):
            #CALCFORWARDKINEMATICS Calculates forward kinematics
            # Inputs:
            #    wr: right wheel speed [rad/s]
            #    wl: left wheel speed  [rad/s]
            # Outputs:
            #    r_dxi : [vx; 0 ; wz] robot velocity vector in robot base frame                        
            #       vx: robot speed in robot frame [m/s]
            #       wz: robot angular vel in robot frame [rad/s]            
            
            vx = 0.5 * self.WheelRadius * (wl+wr)
            wz = (wr-wl) * self.WheelRadius / self.TrackWidth
            
            r_dxi = np.array([vx,0,wz])
            return r_dxi
    def calcInverseKinematics(self, r_dxi):
            #CALCINVERSEKINEMATICS Calculates forward kinematics
            # Inputs:
            #    r_dxi : [vx; 0 ; wz] robot velocity vector in robot base frame             
            #       vx: robot linear speed in robot frame  [m/s]
            #       wz: robot angular speed in robot frame  [rad/s]          
            # Outputs:
            #       wr: right wheel speed [rad/s]
            #       wl: left wheel speed [rad/s]
            
            vx = r_dxi[1]
            wz = r_dxi[3]

            wr = (vx + wz*self.TrackWidth/2.0) / self.WheelRadius
            wl = (vx - wz*self.TrackWidth/2.0) / self.WheelRadius           
            return wr, wl
    
        
    
    @staticmethod
    
    def convTwist2dBaseToWorld(r_dxi, theta):
        #CONVTWIST2DBASETOWORLD  Transforms a twist in robot base frame to world frame
        # Inputs:
        #   - r_dxi: [vx, vy, wz] twist in robot frame [m/s, m/s, rad/s]
        #   - theta: robot orientation [rad]
        # Outpus:
        #   - w_dxi: 2d twist in world frame [m/s, m/s, rad/s]
        R = [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta),  0], 
             [         0,          0,  1]]
        w_dxi = R @ r_dxi;
        return w_dxi
    
    def conv_vec2dBaseToWorld(r_vec,r_xi):
        #CONVTWIST2DBASETOWORLD  Transforms a twist in robot base frame to world frame
        # Inputs:
        #   - r_vec: [x, y,1] vector in robot frame [m/s, m/s, rad/s]
        #   - r_xi: actual pose of robot [x,y,theta]
        # Outpus:
        #   - w_vec: 2d vector in wordl frame
        
        
        R = [[np.cos(r_xi[2]), -np.sin(r_xi[2]), r_xi[0]],
             [np.sin(r_xi[2]), np.cos(r_xi[2]),  r_xi[1]], 
             [         0,          0,  1]]
        
        w_vec = R @ r_vec;
        return w_vec
    
    def step(r_xi, r_dxi, sample_time ):
        #in:  r_xi = [x,y,theta]
        #in:  r_dxi= [vx,0,w]
        #in: sample_time
        tetha =  r_xi[2]+r_dxi[2]*sample_time
        x     =  r_xi[0]+r_dxi[0]*np.cos(r_xi[2])*sample_time
        y     =  r_xi[1]+r_dxi[0]*np.sin(r_xi[2])*sample_time
        
        r_xi_out = np.array([x,y,tetha])
        
        return r_xi_out
        
    
    
    def calc_trayectory(pose_0, r_dxi, time_total, sample_time  ):
        #SIM Simulate trajectories from a given pose
        # This is a vectorized implementation so it can simulate
        # multiple velocity commands. See each input for more details
        # Inputs:
        #   r_dxi: velocity input tensor. This is a matrix of size
        #       [3, numSteps, numCmds] where numSteps is the number of
        #       simulation steps and numCmds is the number of commands.
        #   pose_0: initial pose of the robot [x; y; theta]
        #       It is a column vector of size [3, 1]
        #   sampleTime: simulation sample time [s]. Size [1]
        # Outputs
        #   traj: trajectory tensor. This is a matrix of size [3, numSteps, numCmds]
        
         tiempo = np.arange(0,time_total,sample_time)
         r_xi = np.zeros_like(r_dxi)
         r_xi[0,0,:]=pose_0[0]
         r_xi[0,1,:]=pose_0[1]
         r_xi[0,2,:]=pose_0[2]
         for i in range(len(tiempo)-1):
             r_xi[i+1,2,:]=  r_xi[i,2,:]+r_dxi[i,2,:]*sample_time
             r_xi[i+1,0,:]=  r_xi[i,0,:]+r_dxi[i,0,:]*np.cos(r_xi[i,2,:])*sample_time
             r_xi[i+1,1,:]=  r_xi[i,1,:]+r_dxi[i,0,:]*np.sin(r_xi[i,2,:])*sample_time
             
         return r_xi
    

        
                
# This class will plot the robot with the map
class grafica_robot():
    def __init__(self, modelo_robot, local_planing, mapa):
        size_window = mapa.meter2grid(mapa.size_map)
        self.obj_mapa  = mapa
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.map= np.zeros((size_window[1],size_window[0],3),np.uint8)
        self.image_path= np.zeros((size_window[1],size_window[0],3),np.uint8)
        self.mask_path = np.ones((size_window[1],size_window[0],3),np.uint8)*255
        self.r_xi = np.array([0,0,0])
        self.modelo = modelo_robot
        self.resolucion  = mapa.resolucion ## pixeles/metro
        self.flag_path = False
        self.local_planing =local_planing
        self.size_window = size_window
        self.mapa_2 = np.zeros((size_window[1],size_window[0],3),np.uint8)
        
    def reset_mapa(self):
        self.map = cv2.merge((self.obj_mapa.mapa_inflado,self.obj_mapa.mapa_inflado,self.obj_mapa.mapa_inflado))//4
        self.map = cv2.bitwise_or(self.map,cv2.merge((self.obj_mapa.mapa,self.obj_mapa.mapa,self.obj_mapa.mapa)))
        self.map = cv2.bitwise_and(self.map,self.mask_path)
        self.map = cv2.bitwise_or(self.map,self.image_path)
        

        
    def draw_local_path(self):
        path = self.obj_mapa.world2grid(self.local_planing.local_path)
        for i in range(path.shape[0]-1):
            cv2.line(self.map, path[i,:], path[i+1,:], [0,128,255], 2)
            cv2.circle(self.map, (path[i,:]), 3, [0,128,255], -1)
            cv2.circle(self.map, (path[i+1,:]),3, [0,128,255], -1)
        
    def preprosesing_path(self):
        path = self.obj_mapa.world2grid((self.local_planing.path))

        print(path[:10,:])

        image_path = np.zeros((self.size_window[1],self.size_window[0],3),np.uint8)
        for i in range(path.shape[0]-1):
            cv2.line(image_path, path[i,:], path[i+1,:], [0,0,255], 2)
            cv2.circle(image_path, (path[i,:]), 3, [0,0,255], -1)
            cv2.circle(image_path, (path[i+1,:]), 3, [0,0,255], -1)
            
        self.mask_path = cv2.bitwise_not(cv2.merge((image_path[:,:,2],image_path[:,:,2],image_path[:,:,2])))
        self.image_path = image_path
            
    
    #def imprimir_costos(self):
        
        
        
        
    def graficar_robot(self):
        
        mapa = self.obj_mapa
    
        cv2.circle(self.map, mapa.world2grid(self.r_xi[0:2]), mapa.meter2grid(self.modelo.ChassisRadius), [164,203,255], -1)
        
        x = self.modelo.WheelRadius
        y = self.modelo.ChassisRadius
        start_point = mapa.world2grid(DiffDrive.conv_vec2dBaseToWorld(np.array([x,y,1]) ,self.r_xi)[0:2])
        x = -x
        end_point = mapa.world2grid(DiffDrive.conv_vec2dBaseToWorld(np.array([x,y,1]) ,self.r_xi)[0:2])
        cv2.line(self.map,(start_point[:2]), (end_point[:2]), [255,0,0], 2)
        
        x = self.modelo.WheelRadius
        y = -self.modelo.ChassisRadius
        start_point = mapa.world2grid(DiffDrive.conv_vec2dBaseToWorld(np.array([x,y,1]) ,self.r_xi)[0:2])
        x = -x
        end_point = mapa.world2grid(DiffDrive.conv_vec2dBaseToWorld(np.array([x,y,1]) ,self.r_xi)[0:2])
        cv2.line(self.map, (start_point[:2]),(end_point[:2]), [255,0,0], 2)
        
        x = 0.1
        y = 0
        punto = mapa.world2grid(DiffDrive.conv_vec2dBaseToWorld(np.array([x,y,1]) ,self.r_xi)[0:2])
        cv2.circle(self.map,punto[:2],np.uint32(0.05*self.resolucion), [0,0,255], -1 )
        
    
    def imprimir_mapa(self):
        self.reset_mapa()
        self.draw_local_path()
        self.graficar_robot()
        
        #cv2.imshow("imagen",np.flip((self.map),0))
        #imprimir_image("imagen",np.flip(self.map,0),2)
        imprimir_image("imagen",self.map,2)
        
    def set_pose_robot(self,r_xi):
        self.r_xi= r_xi
        

    def imprimir_trayectorias(self,traj= None):
        self.imprimir_mapa()


# This class will control the map        
class control_map():
    def __init__(self, modelo_robot,resolucion ,size_map, center_map):
        
        # size map  [m]
        # center map [m]
        
        self.modelo_robot = modelo_robot
        self.resolucion = resolucion
        self.size_map = np.array([size_map[0],size_map[1]])
        self.center_map = np.array([center_map[0],center_map[1]])

        #print(self.center_map, self.size_map,self.resolucion )



    def set_mapa(self, mapa):
        self.mapa =(cv2.multiply(mapa,3))
        tam = self.meter2grid(self.modelo_robot.ChassisRadius*2+0.1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(tam,tam))
        self.mapa_inflado =  cv2.dilate(self.mapa,kernel,iterations = 1)
        
    def checkcupancy(self, coordenates):
        # coordenate in x and y in meters [m] is a vector n,2
        coordenates = self.world2grid(coordenates)
        coordenates[:,0]= np.min([coordenates[:,0],( self.mapa.shape[1]-1)*np.ones(coordenates[:,0].shape[0])],axis=0)
        coordenates[:,1]= np.min([coordenates[:,1], (self.mapa.shape[0]-1)*np.ones(coordenates[:,1].shape[0])],axis=0)
        coordenates[:,0]= np.max([coordenates[:,0],np.zeros(coordenates[:,0].shape[0])],axis=0)
        coordenates[:,1]= np.max([coordenates[:,1],np.zeros(coordenates[:,1].shape[0])],axis=0)
        
        
        if(np.any(self.mapa_inflado[coordenates[:,1],coordenates[:,0]])):
            return np.inf
        else:
            return 1
    
    def world2grid(self, coordenate):
        # coordenate contain pos x an y in meters

        coordenate_grid = np.int32((coordenate+self.center_map)*self.resolucion)
        return coordenate_grid
    
    def meter2grid(self, delta):
        # Unlike world2grid this method just calculete with the resolution
        delta_grid = np.int32(delta*self.resolucion)
        return delta_grid


def imprimir_image(nombre,imagen,factor):
    if(len(imagen.shape)>2):
        imageOut=imagen[::factor, ::factor,:]
    else:
        imageOut=imagen[::factor, ::factor]
    cv2.imshow(nombre,imageOut)
    

# Read the map from an image
image_mapa = cv2.imread("mapa/mapa_laberinto_2.png")
# Change the color space of the imga
image_mapa = cv2.cvtColor(image_mapa, cv2.COLOR_BGR2GRAY)

# Read the features of the map
with open('mapa/mapa_laberinto_2.json') as file:
    data = json.load(file)

# Calculate parameters for the map
resolucion = data["datos_mapa"][0]["resolucion"]
pos_x_map = data["datos_mapa"][0]["pos_x_mapa"]/resolucion
pos_y_map = data["datos_mapa"][0]["pos_y_mapa"]/resolucion

# Initialize the map class
mapa = control_map(modelo_turtle(),resolucion,(image_mapa.shape[1]/resolucion,image_mapa.shape[0]/resolucion),(pos_x_map,pos_y_map))
mapa.set_mapa(image_mapa)

# Initialize the trayectory roll out class
traj = TrajRollout(20,20,modelo_turtle().LinVelLims,modelo_turtle().AngVelLims,[-2.5, 2.5],[-np.pi,np.pi],1, 10**-2,1,mapa)
traj.load_path("mapa/path_lab1.txt")

# Interpolation of the path
traj.path_interpolation(3)

# Plot of robot and map
grafica = grafica_robot(modelo_turtle(),traj,mapa)
grafica.preprosesing_path()


# Initialization of robot pose
r_xi = np.array([traj.path[0,0],traj.path[0,1],np.arctan2((traj.path[1,1]-traj.path[0,1]),(traj.path[1,0]-traj.path[0,0]))])
#print(r_xi)
r_dxi = np.array([0,0,0])
grafica.set_pose_robot(r_xi)


# Simulation of robot
while(True):
    try:
        r_dxi = traj.calc_step(r_dxi, r_xi)

        r_xi = DiffDrive.step(r_xi, r_dxi, 30/1000)

        #print(r_xi)
        
        grafica.set_pose_robot(r_xi)
        grafica.imprimir_trayectorias(traj.trayectorias)
        

        key = cv2.waitKey(30) & 0xFF
        if key == ord("z"):
            cv2.destroyAllWindows()
            break
    except Exception as e:
        print(e)
        break


cv2.waitKey(0)

