from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

import numpy as np
from numpy.linalg import norm


class Krum(Strategy):
    def __init__(
            self,
            fraction_fit: float = 0.1,
            fraction_eval: float = 0.1,
            min_fit_clients: int = 2,
            min_eval_clients: int = 2,
            min_available_clients: int = 2,
            eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Weights] = None,
    ) -> None:
        """Federated Averaging strategy.
        Implementation based on https://dl.acm.org/doi/abs/10.5555/3294771.3294783
        Args:
            fraction_fit (float, optional): Fraction of clients used during
                training. Defaults to 0.1.
            fraction_eval (float, optional): Fraction of clients used during
                validation. Defaults to 0.1.
            min_fit_clients (int, optional): Minimum number of clients used
                during training. Defaults to 2.
            min_eval_clients (int, optional): Minimum number of clients used
                during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total
                clients in the system. Defaults to 2.
            eval_fn (Callable[[Weights], Optional[Tuple[float, float]]], optional):
                Function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], Dict[str, Scalar]], optional):
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], Dict[str, Scalar]], optional):
                Function used to configure validation. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds
                containing failures. Defaults to True.
            initial_parameters (Weights, optional): Initial global model parameters.
        """
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Weights]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
        pass

    def configure_fit(self, rnd: int, weights: Weights, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]) -> \
            Optional[Weights]:
        if not results:
            return None
            # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None
            # Convert results
        weights_results = [parameters_to_weights(fit_res.parameters) for client, fit_res in results]
        n_clients: int = len(weights_results)
        matrix: List = []
        for cid in range(0, n_clients):
            client_weights: np.ndarray = np.concatenate([w.flatten() for w in weights_results[cid]])
            matrix.append(client_weights)
        matrix_weights: np.ndarray = np.array(matrix)
        krum_idx = Krum.get_krum_idx(matirx=matrix_weights, n_clients=n_clients, n_malicious=0)
        return weights_results[krum_idx]

    def configure_evaluate(self, rnd: int, weights: Weights, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, rnd: int, results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException]) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None
        return weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss, evaluate_res.accuracy)
                for _, evaluate_res in results
            ]
        )

    @staticmethod
    def get_krum_idx(matirx: np.ndarray, n_clients: int, n_malicious: int) -> int:
        n = n_clients
        f = n_malicious

        def get_krum_dict() -> Dict[int, List[int]]:
            krum_dict: Dict[int, List[int]] = {}
            for i_idx, i in enumerate(matirx):
                current_list: List[Tuple[int, np.ndarray]] = []
                for j_idx, j in enumerate(matirx):
                    if i_idx == j_idx:
                        continue
                    else:
                        current_list.append((j_idx, norm(matirx[i_idx] - matirx[j_idx], ord=1)))
                current_list.sort(key=lambda x: x[1])
                krum_dict[i_idx] = [a[0] for a in current_list]
            return krum_dict

        def get_score(key) -> int:
            total: int = n - f - 2
            result: int = 0
            krum_dict = get_krum_dict()
            for idx in krum_dict[key][:total]:
                result += norm(matirx[key] - matirx[idx], ord=1) ** 2
            return result

        clients_score = [(i, get_score(i)) for i in range(0, n_clients)]
        return min(clients_score, key=lambda x: x[1])[0]
