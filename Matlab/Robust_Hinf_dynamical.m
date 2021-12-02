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
W_i = tf([1],[1,0]); %weight tracking performance 
W_track = 0.001*tf([1],[1/10,1]);
W_u = 0.01*tf([1 0],[1/10,1]) * tf([1],[1/1000,1]); %penalize controller gains
W_r = 1;
W_qhat = 1;
C_q1 = [1 0];
C_q2 = [0 1]; 
wim = Info3.W1;
wim2 = Info4.W1;
Plant_noise = [wim , 0.1*wim;
               0.1*wim , wim];
           
Output_Uncertainty = [wim2 , 0;
               0, wim2];

% Interconnection:
systemnames = 'Plant_nom Plant_noise Output_Uncertainty W_track W_u W_i W_r W_qhat C_q1 C_q2';
inputvar = '[reward;Q2;u]';
outputvar = '[W_track;W_u;W_i;C_q1;C_q2]';
input_to_Plant_nom = '[u;W_r]';
input_to_Plant_noise = '[Plant_nom]';
input_to_Output_Uncertainty = '[Plant_nom]';
input_to_W_track = '[W_r+W_qhat-C_q1]';
input_to_W_i = '[W_r+W_qhat-C_q1]';
input_to_W_u = '[u]';
input_to_C_q1 = '[Plant_nom+Output_Uncertainty+Plant_noise]';
input_to_C_q2 = '[Plant_nom+Output_Uncertainty+Plant_noise]';
input_to_W_r = '[reward]';
input_to_W_qhat = '[Q2]';
cleanupsysic = 'yes';
PK = sysic;

K1 = []
while isempty(K1)
% Synthesize two-degree of freedom controller.
nmeas = 3;					% # of measurements
ncont = 1;					% # of controls
[K1,CL,GAM,INFO] = hinfsyn(PK, nmeas, ncont);
disp('Trying to compute controller...')
end
[K1,g] = balreal(K1);  % Compute balanced realization
elim = (g<1e-2);         % Small entries of g are negligible states
rK = modred(K1,elim) % Remove negligible states
GAM
step(CL)

sigma(CL,ss(GAM))
xlim([1e-5,1e5]);
