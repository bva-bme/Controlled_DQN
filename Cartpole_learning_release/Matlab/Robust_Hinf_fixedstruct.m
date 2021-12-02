close all
clear all
clc

%Parameters
gamma = 1;
Theta1_nom = 220;
Theta2_nom = 222;
Theta1_min = 200;
Theta1_max = 500;

% Define the time-varying parameters. 
rho1 = ureal('rho1',1,'Range',[Theta1_min / Theta1_nom, Theta1_max / Theta1_nom]); 
rho2 = rho1*ureal('corr', 1, 'Range', [0.8, 1.2]);

% System dynamics
A = [-Theta1_nom * rho1, gamma * Theta1_nom * rho1; -Theta2_nom*rho2, gamma * Theta2_nom * rho2];
B = [Theta1_nom * rho1, Theta1_nom * rho1; 0, Theta2_nom * rho2];
C = [1 0; 0 1];
D = [1e-10 0; 0 0];
Plant = ss(A, B, C, D);

samples = 200;
sampled_Plant = usample(Plant, samples);
Plant_nom = Plant.NominalValue;
[Plant_noise, Info3]= ucover(sampled_Plant,Plant_nom,2,'OutputMult'); %Create a bound for the uncertain system

Q1_noise = ureal('Q1_noise', 0, 'Range', [-0.2, 0.2]);
Q2_noise = ureal('Q2_noise', 0, 'Range', [-0.5, 0.5]);
Output_U = ss([Q1_noise 0; 0 Q2_noise]);
[Output_Uncertainty, Info4]= ucover(Output_U,ss([1, 0; 0, 1]),1,'OutputMult'); %Create a bound for the uncertain system

% Weights and helpers
W_track = 0.00001*tf([1],[1/10,1]);
W_u = 0.001*tf([1 0],[1/10,1]) * tf([1],[1/1000,1]); %penalize controller gains
W_r = tf([1],[1/5,1]) * tf([1 0],[1/1000,1]);
W_qhat = tf([1],[1/5,1]) * tf([1 0],[1/1000,1]);
C_q1 = [1 0];
C_q2 = [0 1]; 
wim = Info3.W1;
wim2 = Info4.W1;
Plant_noise = [wim , 0.1*wim;
               0.1*wim , wim];
           
Output_Uncertainty = [wim2 , 0;
               0, wim2];

% Controller
s=tf('s');
Ki = realp('Ki', 0.937)/s;
Kq1 = realp('Kq1', -0.429);
Kq2 = realp('Kq2', 0.751);
K = [Kq1 Kq2];

% Interconnection:
systemnames = 'Plant_nom Plant_noise Output_Uncertainty K Ki W_track W_u W_r W_qhat C_q1 C_q2';
inputvar = '[reward;Q2]';
outputvar = '[W_track;W_u]';
input_to_Plant_nom = '[Ki+K;W_r]';
input_to_Plant_noise = '[Plant_nom]';
input_to_Output_Uncertainty = '[Plant_nom]';
input_to_W_track = '[W_r+W_qhat-C_q1]';
input_to_W_u = '[Ki+K]';
input_to_K = '[Plant_nom+Output_Uncertainty+Plant_noise]';
input_to_Ki = '[reward+C_q2-C_q1]';
input_to_C_q1 = '[Plant_nom+Output_Uncertainty+Plant_noise]';
input_to_C_q2 = '[Plant_nom+Output_Uncertainty+Plant_noise]';
input_to_W_r = '[reward]';
input_to_W_qhat = '[Q2]';
cleanupsysic = 'yes';
PK = sysic;

%Design
opts = hinfstructOptions('RandomStart',5)
[K1,CL,GAM,INFO] = hinfstruct(PK, opts)
showBlockValue(K1)

%Plot results
step(PK)
figure
sigma(PK)
xlim([1e-8,1e5]);
