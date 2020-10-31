# pyro-tutorial-code
GP Regression pyro code from tutorial

When I restart the jupyter kernel and run all blocks of codes, I get the same results in my question yesterday(GP code 2.ipynb);
However, after that, I rerun the specific training code alone:
optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
num_steps = 2500 if not smoke_test else 2
for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(gpr.model, gpr.guide)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
  
I get the expected results in GP code.ipynb......
