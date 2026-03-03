import random
from simulation_pkg import basic

def main():
    # 주차장에 주차된 차 3대
    basic.load_model( "obstacle4", basic.model_types[3] , basic.obstacle_coordinates_1 ) # 주차공간 1번
    #basic.load_model( "obstacle5", basic.model_types[4] , basic.obstacle_coordinates_2 ) # 주차공간 2번
    basic.load_model( "obstacle6", basic.model_types[5] , basic.obstacle_coordinates_3 ) # 주차공간 3번
    #basic.load_model( "obstacle7", basic.model_types[5] , basic.obstacle_coordinates_4 ) # 주차공간 4번
    
    # 도로에 있는 차 3대
    #basic.load_model( "obstacle1", basic.model_types[2] , basic.obstacle_coordinates1 )
    #basic.load_model( "obstacle2", basic.model_types[0] , basic.obstacle_coord(basic.obstacle_coordinates2))
    #basic.load_model( "obstacle3", basic.model_types[1] , basic.obstacle_coord(basic.obstacle_coordinates3))

if __name__ == "__main__":
    main()