import stim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
import heapq


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, circuit, dem, err_costs, hidden_dim=100, depth=3, epochs=20, batch_size=128, lr=1e-3, lr_gamma=0.5, use_cuda=True, train_shots_per_epoch=1000, test_shots_per_epoch=1000):
        self.circuit = circuit
        self.dem = dem
        self.err_costs = err_costs
        self.input_dim = dem.num_detectors
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.train_shots_per_epoch = train_shots_per_epoch
        self.test_shots_per_epoch = test_shots_per_epoch

    def _build_model(self):
        layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.depth - 1):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(self.hidden_dim, 1)]
        return nn.Sequential(*layers)

    def fit(self, X_unused=None, y_unused=None):
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = self._build_model().to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=self.lr_gamma)
        criterion = nn.MSELoss()

        # Fixed test set for evaluation
        test_dets, _, test_errs = self.dem.compile_sampler(seed=424242).sample(shots=self.test_shots_per_epoch, return_errors=True)
        X_test = test_dets.astype(np.float32)
        y_test = np.array([np.dot(err, self.err_costs) for err in test_errs])

        self.model.train()
        for epoch in range(self.epochs):
            print(f"\n[Epoch {epoch+1}/{self.epochs}] Sampling training data...")
            dets, _, errs = self.dem.compile_sampler(seed=23934573 + epoch).sample(shots=self.train_shots_per_epoch, return_errors=True)
            X = dets.astype(np.float32)
            y = np.array([np.dot(err, self.err_costs) for err in errs])
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(device)
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            print(f'        Training')
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            scheduler.step()  # Advance LR scheduler

            # Evaluation on test set
            self.model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)
                preds_test = self.model(X_test_tensor).cpu().numpy().flatten()
            mse = mean_squared_error(y_test, preds_test)
            r2 = r2_score(y_test, preds_test)
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Epoch {epoch+1}] LR: {current_lr:.5e}, Test MSE: {mse:.4f}, Test R²: {r2:.4f}")

        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        with torch.no_grad():
            return self.model(X_tensor.to(device)).cpu().numpy().flatten()

    def estimate_cost(self, dets):
        self.model.eval()
        dets = np.asarray(dets, dtype=np.float32).reshape(1, -1)
        return self.predict(dets)[0]



circuit_fname = 'testdata/colorcodes/r=5,d=5,p=0.002,noise=si1000,c=superdense_color_code_X,q=37,gates=cz.stim'
circuit = stim.Circuit.from_file(circuit_fname)
dem = circuit.detector_error_model(decompose_errors=False)

err_costs = []
num_observables = dem.num_observables
err_activates_detectors = []
err_activates_obs = []
for instruction in dem.flattened():
    if instruction.type != 'error':
        continue
    p = instruction.args_copy()[0]
    err_costs.append(-np.log(p / (1 - p)))
    activated_dets = set()
    activated_obs = set()
    for target in instruction.targets_copy():
        if target.is_separator():
            continue
        d = target.val
        if target.is_logical_observable_id():
          if d in activated_obs:
              activated_obs.remove(d)
          else:
              activated_obs.add(d)
        else:
          assert target.is_relative_detector_id()
          if d in activated_dets:
              activated_dets.remove(d)
          else:
              activated_dets.add(d)
    err_activates_detectors.append(activated_dets)
    err_activates_obs.append([1 if i in activated_obs else 0 for i in range(num_observables)])

err_activates_obs = np.array(err_activates_obs)
detector_to_errors = [[] for _ in range(dem.num_detectors)]
for e, dets_set in enumerate(err_activates_detectors):
    for d in dets_set:
        detector_to_errors[d].append(e)

err_costs = np.array(err_costs)

model = TorchMLPRegressor(
    circuit=circuit,
    dem=dem,
    err_costs=err_costs,
    hidden_dim=10 * dem.num_detectors,
    depth=10,
    epochs=200,
    lr=1e-3,
    lr_gamma=0.9,
    train_shots_per_epoch=10000,
    test_shots_per_epoch=1000,
)
model.fit()

# Sample a fresh test set
dets_test, _, errs_test = dem.compile_sampler(seed=12345).sample(shots=2000, return_errors=True)
X_test = dets_test.astype(np.float32)
y_test = np.array([np.dot(err, err_costs) for err in errs_test])

# Predict and compute statistics
test_preds = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)

print("Torch MLP Model Performance on Fresh Test Set:")
print(f"  Test MSE: {test_mse:.4f}")
print(f"  Test R^2: {test_r2:.4f}")
print()


factor = 1.0
# ----- A* Decoder -----
def decode(dets, model, err_activates_detectors, detector_to_errors, err_costs, num_detectors):
    dets = np.asarray(dets, dtype=bool)
    initial_cost = model.estimate_cost(dets) * factor
    frontier = [(initial_cost, 0.0, [], dets.copy(), np.zeros(len(err_activates_detectors), dtype=bool))]

    while frontier:
        total_est_cost, base_cost, errors_used, rem_dets, blocked = heapq.heappop(frontier)
        print(f'total_est_cost = {total_est_cost} errors_used = {errors_used} rem_dets = {np.where(rem_dets)[0]} est_rem_cost = {total_est_cost-base_cost}')
        rem_det_indices = np.flatnonzero(rem_dets)
        if len(rem_det_indices) == 0:
            result = np.zeros(len(err_activates_detectors), dtype=bool)
            for e in errors_used:
                result[e] = True
            return result

        d = rem_det_indices[0]
        for e in detector_to_errors[d]:

            new_errors_used = errors_used + [e]
            new_rem_dets = rem_dets.copy()
            for j in err_activates_detectors[e]:
                new_rem_dets[j] ^= True

            new_blocked = blocked.copy()
            for other_e in detector_to_errors[d]:
                if other_e == e:
                    break
                new_blocked[other_e] = True

            new_base_cost = base_cost + err_costs[e]
            est_cost = model.estimate_cost(new_rem_dets) * factor
            heapq.heappush(frontier, (new_base_cost + est_cost, new_base_cost, new_errors_used, new_rem_dets, new_blocked))

    raise RuntimeError("A* search failed to find a decoding path.")


# Evaluate A* decoder
sampler = dem.compile_sampler(seed=999)
dets, obs, errs = sampler.sample(shots=100, return_errors=True)

num_weight_leq_truth = 0
total = 0
num_errors = 0
for shot in range(len(dets)):
    actual_cost = np.dot(errs[shot], err_costs)
    print(f'actual cost = {actual_cost}')
    print(f'actual errs = {np.where(errs[shot])}')
    decoded = decode(dets[shot], model, err_activates_detectors, detector_to_errors, err_costs, dem.num_detectors)
    decoded_cost = np.dot(decoded, err_costs)
    true_cost = np.dot(errs[shot], err_costs)
    if true_cost >= decoded_cost:
        num_weight_leq_truth += 1
    decoded_obs = np.dot(decoded, err_activates_obs) % 2
    if (decoded_obs != obs[shot]).any():
      num_errors += 1
    total += 1
    print(f'total = {total} num_weight_leq_truth = {num_weight_leq_truth} num_errors = {num_errors}')

print(f"A* decoder had logical errors on {num_errors}/{total} shots ({100 * num_errors / total:.2f}%)")
print(f"A* decoder matched total cost on {num_weight_leq_truth}/{total} shots ({100 * num_weight_leq_truth / total:.2f}%)")
