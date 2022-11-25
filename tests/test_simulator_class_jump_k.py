from langesim import Simulator
import numpy as np
import pytest

def test_jump_k():
    """Test that if the stifness jumps from ki to ko at t=0, the variance
    evolves according to
    <x(t)^2> = exp(-2ko t)/ki + (1-exp(-2ko t))/ko
    """
    ki = 1.0 
    ko = 3.0 

    def theo_var(t):
        """Theoretical variance"""
        return np.exp(-2.0*ko*t)/ki + (1.0-np.exp(-2.0*ko* t))/ko

    def center(t):
        """Center of the harmonic potential fixed at 0"""
        return 0.0

    def k(t):
        """Stifness of the harmonic potential: jumps from ki to ko at t=0"""
        if t <= 0.0:
            return ki 
        else:
            return ko 
    
    tot_sims=100_000
    simulator = Simulator(
        tot_sims = tot_sims,
        dt = 0.001, 
        tot_steps = 1000,
        snapshot_step = 100, 
        k = k,
        center = center
    )

    simulator.run()
    sim = simulator.simulation[0]
    tol = 4.7 / np.sqrt(tot_sims)
    sim.build_variances('x')
    for t in sim.results['times']:
        var = theo_var(t)
        assert sim.variance_func['x'](t) == pytest.approx(var, abs = var * tol), f"variance different from theory at t={t}"
    

    