# Gas-compressor-failure-prediction
Gas compressor failure prediction using "sparse auto encoder" trained weights and deployment using tensorflow serving.


### Plan of attack:-
* Trained sparse auto encoder on the gas compressor data.
* Used the trained weights of the auto encoder to initialize the weights of the fully connected NN.
* Sparse auto encoder architecture 25,50,100,50,25.
* Fully connected NN architecture 25,50,100,50,15.

### Model deployment:-
* Exported the model using SavedModelBuilder class of the tensorflow.
* Used **tensorflow_model_server** which hosts the model on the server.
* Created a flask server which listens to the client requests and queries the tensorflow_model_server for inference.
* Communication between the flask and tensorflow model server takes place with the help of gRPC calls written in translate_client.py file.

### Gas Compressor Dataset:-
A 25-feature vector containing the vessel relevant features:
* Lever (lp) 
* Speed [knots]
* Gas Turbine shaft torque (GTT) [kN m]
* Gas Turbine Speed (GT rpm) [rpm]
* Controllable Pitch Propeller Thrust stbd (CPP T stbd)[N]
* Controllable Pitch Propeller Thrust port (CPP T port)[N]
* Shaft Torque port (Q port) [kN]
* Shaft rpm port (rpm port)[rpm]
* Shaft Torque stbd (Q stdb) [kN]
* Shaft rpm stbd (rpm stbd) [rpm]
* HP Turbine exit temperature (T48) [C]
* Generator of Gas speed (GG rpm) [rpm]
* Fuel flow (mf) [kg/s]
* ABB Tic control signal (ABB Tic)
* GT Compressor outlet air pressure (P2) [bar]
* GT Compressor outlet air temperature (T2) [C]
* External Pressure (Pext) [bar]
* HP Turbine exit pressure (P48) [bar]
* TCS tic control signal (TCS tic) 
* Thrust coefficient stbd (Kt stbd) 
* Propeller rps stbd (rps prop stbd) [rps]
* Thrust coefficient port (Kt port) 
* Propeller rps port (rps prop port) [rps]
* Propeller Torque port (Q prop port) [Nm]
* Propeller Torque stbd (Q prop stbd) [Nm]

**Target Label is "GT Compressor decay state coefficient (KMcompr)" which takes 15 different values, equivalent to measuring the current condition of the machine.**
