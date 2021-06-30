import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


from api import b0RemoteApi
from car import Car
from meta import MetaManager
from driving import Driver
from time import time


CLIENT_NAME = 'Python_client'
CHANNEL_NAME = 'b0RemoteApi'


with b0RemoteApi.RemoteApiClient(CLIENT_NAME, CHANNEL_NAME, inactivityToleranceInSec=60, timeout=30) as client:
    car = Car(client)
    mm = MetaManager(client)
    driver = Driver(car, mm)

    # Signs recognition test
    driver.drive_to_path('Path165', offset=0.5)

    # Straight Parking
    # driver.drive_to_path('Path197', offset=0.4)
    # driver.drive_to_path('Path192', offset=0.4, backward=True)
    # driver.drive_to_path('Path194', offset=0.4)
    # driver.drive_to_path('Path192', offset=0.3, backward=True)
    # driver.drive_to_path('Path195', offset=0.4)
    # driver.drive_to_path('Path192', offset=0.3, backward=True)
    # driver.drive_to_path('Path170', offset=0.5)

    # Skewed parking

    #
    # driver.drive_to_path('Path218', offset=0.5)
    # driver.drive_to_path('Path213', offset=0.0, backward=True)
    # driver.drive_to_path('Path224', offset=0.5)
    # driver.drive_to_path('Path230', offset=0.0, backward=True)
    # driver.drive_to_path('Path152', offset=0.5)

    #Reversing

    #
    # driver.drive_to_path('Path29', offset=0.1)
    # driver.drive_to_path('Path4', offset=0.1, backward=True)
    # driver.drive_to_path('Path41', offset=0.5)

    # # Driving forward to some streets
    # driver.drive_to_structure('StreetPaths9')
    # driver.drive_to_structure('RoundaboutPaths0')
    #
    # # Drive backward to some streets
    # driver.drive_to_structure('StreetPaths9', backward=True)
    # driver.drive_to_structure('RoundaboutPaths0', backward=True)

    while True:
        start = time()
        car.refresh()
        driver.update()
        
        #driver.use_lidar()
        car.apply()
        # print(time() - start, car.orient)
