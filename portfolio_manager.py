from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Position:
    currency: str
    size: float
    entry_value: float


@dataclass
class TradeDecision:
    base: str
    quote: str
    fraction: float
    kelly_fraction: float
    confidence: float
    regime: str


class PortfolioManager:
    def __init__(
        self,
        mode: str = 'single_asset',
        initial_capital: float = 10000.0,
        kelly_fraction: float = 0.25,
        max_position: float = 1.0,
        vol_window: int = 20,
        fee_rate: float = 0.001
    ):
        self.mode = mode
        self.initial_capital = initial_capital
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.vol_window = vol_window
        self.fee_rate = fee_rate
        
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.total_value = initial_capital
        
        self._win_history: Dict[Tuple[str, str], List[float]] = {}
        self._volatility: Dict[Tuple[str, str], List[float]] = {}
        
    def decide(
        self,
        graph,
        holding: str,
        trade: Optional[Tuple[str, str]],
        predicted_accels: Dict[Tuple[str, str], float]
    ) -> Optional[TradeDecision]:
        if trade is None:
            return None
            
        base, quote = trade
        edge = (base, quote)
        
        if self.mode == 'single_asset':
            return TradeDecision(
                base=base,
                quote=quote,
                fraction=1.0,
                kelly_fraction=1.0,
                confidence=1.0,
                regime='single_asset'
            )
        
        conf = self._compute_confidence(edge, predicted_accels)
        kelly = self._compute_kelly(edge)
        regime = self._detect_regime(edge, graph)
        
        fraction = min(kelly * conf, self.max_position)
        
        return TradeDecision(
            base=base,
            quote=quote,
            fraction=fraction,
            kelly_fraction=kelly,
            confidence=conf,
            regime=regime
        )
    
    def rank_all_flows(
        self,
        graph,
        predicted_accels: Dict[Tuple[str, str], float]
    ) -> List[TradeDecision]:
        """Rank all flows - long winners AND short losers."""
        decisions = []
        
        for edge, pred_accel in predicted_accels.items():
            if abs(pred_accel) < 0.0001:
                continue
            
            base, quote = edge
            conf = self._compute_confidence(edge, predicted_accels)
            kelly = self._compute_kelly(edge)
            regime = self._detect_regime(edge, graph)
            
            if pred_accel > 0:
                decisions.append(TradeDecision(
                    base=base,
                    quote=quote,
                    fraction=1.0,
                    kelly_fraction=kelly,
                    confidence=conf,
                    regime=regime
                ))
            elif pred_accel < 0:
                decisions.append(TradeDecision(
                    base=quote,
                    quote=base,
                    fraction=1.0,
                    kelly_fraction=kelly,
                    confidence=conf,
                    regime=regime
                ))
        
        decisions.sort(key=lambda d: abs(d.kelly_fraction * d.confidence), reverse=True)
        return decisions
    
    def _compute_confidence(
        self,
        edge: Tuple[str, str],
        predicted_accels: Dict[Tuple[str, str], float]
    ) -> float:
        pred = predicted_accels.get(edge, 0.0)
        abs_pred = abs(pred)
        
        conf = np.clip(abs_pred * 100.0, 0.0, 1.0)
        
        return conf
    
    def _compute_kelly(self, edge: Tuple[str, str]) -> float:
        if edge not in self._win_history:
            return self.kelly_fraction
            
        history = self._win_history[edge]
        if len(history) < 10:
            return self.kelly_fraction
            
        wins = [w for w in history if w > 0]
        losses = [w for w in history if w <= 0]
        
        if not wins or not losses:
            return self.kelly_fraction
            
        win_rate = len(wins) / len(history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return self.kelly_fraction
            
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b
        
        kelly = np.clip(kelly * self.kelly_fraction, 0.0, self.max_position)
        
        return kelly
    
    def _detect_regime(
        self,
        edge: Tuple[str, str],
        graph
    ) -> str:
        if edge not in graph.edge_state:
            return 'neutral'
            
        es = graph.edge_state[edge]
        
        if edge not in self._volatility:
            self._volatility[edge] = []
            
        vol = abs(es.velocity)
        self._volatility[edge].append(vol)
        
        if len(self._volatility[edge]) > self.vol_window:
            self._volatility[edge].pop(0)
        
        if len(self._volatility[edge]) < 5:
            return 'neutral'
            
        rolling_vol = np.std(self._volatility[edge])
        mean_vol = np.mean(self._volatility[edge])
        
        if rolling_vol > mean_vol * 1.5:
            return 'high_vol'
        elif rolling_vol < mean_vol * 0.5:
            return 'low_vol'
        else:
            return 'normal'
    
    def execute(
        self,
        decision: TradeDecision,
        graph,
        bar_idx: int
    ) -> Tuple[float, str]:
        if decision.fraction <= 0:
            return 0.0, self._get_holding(graph)
            
        base = decision.base
        quote = decision.quote
        
        current_value = self.cash
        trade_value = current_value * decision.fraction
        
        if quote in self.positions:
            trade_value += self.positions[quote].size * self.positions[quote].entry_value
        
        fee = trade_value * self.fee_rate
        trade_value_after_fee = trade_value - fee
        
        self.cash = current_value - trade_value_after_fee
        
        if quote not in self.positions:
            self.positions[quote] = Position(
                currency=quote,
                size=0.0,
                entry_value=0.0
            )
        
        self.positions[quote].size = trade_value_after_fee
        self.positions[quote].entry_value = 1.0
        
        return trade_value_after_fee, quote
    
    def record_pnl(self, edge: Tuple[str, str], pnl: float):
        if edge not in self._win_history:
            self._win_history[edge] = []
            
        self._win_history[edge].append(pnl)
        
        if len(self._win_history[edge]) > 100:
            self._win_history[edge].pop(0)
    
    def get_holding(self, graph) -> str:
        if not self.positions:
            return 'USD'
            
        largest_pos = max(
            self.positions.items(),
            key=lambda x: x[1].size if x[1].size > 0 else 0
        )
        
        if largest_pos[1].size > 0:
            return largest_pos[0]
            
        return 'USD'
    
    def _get_holding(self, graph) -> str:
        return self.get_holding(graph)
    
    def get_portfolio_value_usd(self, graph) -> float:
        """Compute total portfolio value in USD via shortest path to USD."""
        total_usd = self.cash
        
        for currency, pos in self.positions.items():
            if pos.size <= 0:
                continue
            
            paths = graph.dijkstra(currency)
            
            if 'USD' in paths:
                cost, path = paths['USD']
                if len(path) >= 2:
                    path_value = pos.size
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i + 1])
                        if edge in graph.edges:
                            df = graph.edges[edge]
                            if len(df) > 0:
                                price = df['close'].iloc[-1]
                                path_value = path_value * price
                    total_usd += path_value
            else:
                total_usd += pos.size
        
        return total_usd
    
    def get_stats(self, graph=None) -> Dict:
        total_pos_value = sum(p.size for p in self.positions.values() if p.size > 0)
        
        stats = {
            'cash': self.cash,
            'positions_value': total_pos_value,
            'total_value': self.cash + total_pos_value,
            'mode': self.mode,
            'n_positions': len([p for p in self.positions.values() if p.size > 0])
        }
        
        if graph is not None:
            stats['total_value_usd'] = self.get_portfolio_value_usd(graph)
        
        return stats
