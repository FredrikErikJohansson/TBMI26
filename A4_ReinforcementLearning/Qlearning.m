%% Initialization
%  Initialize the world, Q-table, and hyperparameters
actions = [1 2 3 4];
probs = [1 1 1 1];
exploration = 0.5;
learnRate = 0.2;
discount = 0.9;
episodes = 1000;
world = 1;
state = gwinit(world);
Q = randn(state.ysize, state.xsize, length(actions));
Q(1,:,2)   = -inf;
Q(end,:,1) = -inf;
Q(:,1,4)   = -inf;
Q(:,end,3) = -inf;

%% Training loop
%  Train the agent using the Q-learning algorithm.
for i = 1:episodes
    disp((i/episodes)*100)
    state = gwinit(world);
    y = state.pos(1,1);
    x = state.pos(2,1);
    while(state.isvalid)
        exploration = getepsilon(i, episodes);
        [a, ~] = chooseaction(Q, y, x, actions, probs, exploration);
        state = gwaction(a);
        y2 = state.pos(1,1);
        x2 = state.pos(2,1);
        y = y2;
        x = x2;
        reward = state.feedback;
        [~, oa] = chooseaction(Q, y2, x2, actions, probs, exploration);
        
        Q(y, x, a) = (1- learnRate)*Q(y, x, a) + learnRate*(reward + discount*Q(y2, x2, oa));  
        
        if(state.isterminal)
            break;
        end
    end
end

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.

for k = 1:4
    figure(k);
    imagesc(Q(:,:,k));
end

figure(5);

numTest = 10;
accuracy = 0;

for i = 1:numTest
    state = gwinit(world);
    y = state.pos(1,1);
    x = state.pos(2,1);
    for j = 1:20
        [a, ~] = chooseaction(Q, y, x, actions, probs, 0);

        state = gwaction(a);
        y = state.pos(1,1);
        x = state.pos(2,1);
        
        if(state.isvalid == false)
            break;
        end
        
        gwdraw(i);

        if(state.isterminal)
            accuracy = accuracy + 1;
            break;
        end
    end
end

accuracy/numTest

