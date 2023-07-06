function M=fcn_get_mIRR_model(Y,y_model)
    yref=mode(Y,2);  
    pp=[1:5,0]; 
    M=fcn_getCM(yref,y_model,pp);
end
