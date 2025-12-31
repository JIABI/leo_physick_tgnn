from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import torch

from .channel import compute_sinr, sinr_to_rate
from .load import update_satellite_load
from .interference import build_satellite_interference_edges
from ..graph.builder import build_user_sat_edges
from ..graph.edge_types import EdgeType
from ..physics.descriptors import compute_edge_descriptors


def _parse_utc(s: str) -> datetime:
    """Parse ISO8601 UTC time string."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sample_points_on_sphere(n: int, radius: float, device: torch.device) -> torch.Tensor:
    v = torch.randn(n, 3, device=device)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
    return v * float(radius)


class MultiUserLEOEnv:
    """Environment that emits time-varying user-satellite graphs.

    Key outputs per step:
      - node_x: [N,6] = [pos(3), vel(3)] (scaled)
      - edge_index / edge_z / edge_type for:
          * user->sat candidate edges (USER_SAT)
          * optional sat<->sat proximity edges (SAT_SAT) when graph.include_sat_sat=True
      - y: supervision (currently satellite load/utilization on sat nodes)
      - meta: system-level signals (serving_sat, ho_fail, handover, load stats)

    Notes:
      - sat_load is maintained as a *utilization* in [0, ~] where 1.0 corresponds to sat_capacity_users users assigned.
      - A simple greedy allocator generates ground-truth serving decisions in multi-user settings.
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device

        self.T = int(cfg.get("T", 200))
        self.dt = float(cfg.get("dt", 1.0))
        self.K = int(cfg.get("K_users", 50))

        torch.manual_seed(int(cfg.get("seed", 7)))

        # ---- Graph params ----
        gcfg = cfg.get("graph", {})
        self.visibility_radius = float(gcfg.get("visibility_radius", 0.7))
        self.topk = int(gcfg.get("topk", 8))

        self.include_sat_sat = bool(gcfg.get("include_sat_sat", False))
        self.sat_sat_radius = float(gcfg.get("sat_sat_radius", 0.25))

        # ---- Channel params ----
        ccfg = cfg.get("channel", {})
        self.base_sinr = float(ccfg.get("base_sinr", 10.0))
        self.dist_scale = float(ccfg.get("dist_scale", 5.0))
        self.sinr_min = float(ccfg.get("sinr_min", -1.0))  # feasibility threshold in surrogate units

        # ---- Handover cost params ----
        hcfg = cfg.get("handover", {})
        self.cost_intra_beam = float(hcfg.get("cost_intra_beam", 0.1))
        self.cost_intra_sat = float(hcfg.get("cost_intra_sat", 0.3))
        self.cost_inter_sat = float(hcfg.get("cost_inter_sat", 1.0))

        # ---- Load / capacity params ----
        lcfg = cfg.get("load", {})
        self.sat_capacity_users = int(lcfg.get("sat_capacity_users", 10))  # users per satellite capacity (debug)
        self.load_momentum = float(lcfg.get("momentum", 0.8))

        # ---- Physics params ----
        self.cox_cfg = cfg.get("cox", {})

        # ---- Scale factors (km -> normalized units) ----
        eph = cfg.get("ephemeris", {})
        self.pos_scale = float(eph.get("pos_scale", 1.0))
        self.vel_scale = float(eph.get("vel_scale", 1.0))

        # ---- Ephemeris selection ----
        mode = str(eph.get("mode", "debug"))
        if mode == "skyfield_tle":
            from .ephemeris import HybridUserSatEphemeris, load_tle_file

            tle_path = str(eph.get("tle_path", ""))
            if not tle_path:
                raise ValueError("ephemeris.mode=skyfield_tle requires ephemeris.tle_path")
            tle_records = load_tle_file(tle_path)

            self.S = len(tle_records)
            self.N = self.K + self.S

            start_time = _parse_utc(str(eph.get("start_time_utc", "2025-12-25T00:00:00Z")))

            place_users = bool(eph.get("place_users_on_earth", True))
            earth_r_km = float(eph.get("user_earth_radius_km", 6371.0))

            if place_users:
                user_pos0_km = _sample_points_on_sphere(self.K, earth_r_km, device)
            else:
                user_pos0_km = torch.randn(self.K, 3, device=device) * (0.25 * earth_r_km)

            # Users are quasi-static in this synthetic setting
            user_vel0_km_s = torch.zeros(self.K, 3, device=device)

            self.ephem = HybridUserSatEphemeris(
                user_pos0=user_pos0_km,
                user_vel0=user_vel0_km_s,
                tle_records=tle_records,
                start_time_utc=start_time,
                device=device,
            )

        else:
            # Debug constant-velocity kinematics in already-scaled units
            from .ephemeris import SimpleKinematicEphemeris

            self.S = int(cfg.get("S_sats", 200))
            self.N = self.K + self.S

            user_pos0 = torch.randn(self.K, 3, device=device) * 0.2
            user_vel0 = torch.randn(self.K, 3, device=device) * 0.01
            sat_pos0 = torch.randn(self.S, 3, device=device) * 0.2
            sat_vel0 = torch.randn(self.S, 3, device=device) * 0.02

            pos0 = torch.cat([user_pos0, sat_pos0], dim=0)
            vel0 = torch.cat([user_vel0, sat_vel0], dim=0)
            self.ephem = SimpleKinematicEphemeris(pos0, vel0)

        # sat_load is utilization: 1.0 ~ sat_capacity_users users assigned
        self.sat_load = torch.zeros(self.S, device=device)
        self.prev_serving_sat = torch.full((self.K,), -1, device=device, dtype=torch.long)
        self.t = 0

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self.sat_load.zero_()
        self.prev_serving_sat.fill_(-1)
        return self._make_step()

    def step(self) -> Dict[str, Any]:
        if self.t >= self.T:
            raise RuntimeError("Episode finished")
        self.t += 1
        return self._make_step()

    def _greedy_assign(
        self,
        src_u: torch.Tensor,        # [E] global user idx (0..K-1)
        dst_s: torch.Tensor,        # [E] local sat idx (0..S-1)
        rate: torch.Tensor,         # [E]
        sinr: torch.Tensor,         # [E]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Greedy allocator that produces serving satellites and per-step system signals."""
        device = rate.device
        K = self.K
        S = self.S

        serving = torch.full((K,), -1, device=device, dtype=torch.long)
        ho_fail = torch.zeros((K,), device=device, dtype=torch.bool)
        handover = torch.zeros((K,), device=device, dtype=torch.bool)

        incoming_users = torch.zeros((S,), device=device, dtype=torch.float32)

        # Slight randomization to reduce deterministic order bias
        user_order = torch.randperm(K, device=device)

        cap = max(1, int(self.sat_capacity_users))
        delta_util = 1.0 / float(cap)

        for u in user_order.tolist():
            m = (src_u == int(u))
            if not torch.any(m):
                ho_fail[u] = True
                continue

            idx = torch.nonzero(m, as_tuple=False).view(-1)
            # Sort candidates by rate descending
            r = rate[idx]
            s = dst_s[idx]
            q = sinr[idx]

            order = torch.argsort(r, descending=True)
            picked = -1
            for j in order.tolist():
                sj = int(s[j].item())
                if float(q[j].item()) < float(self.sinr_min):
                    continue
                # capacity check: current utilization + new assignment <= 1.0
                util = float(self.sat_load[sj].item()) + float(incoming_users[sj].item()) * delta_util
                if util + delta_util <= 1.0 + 1e-6:
                    picked = sj
                    incoming_users[sj] += 1.0
                    break

            if picked < 0:
                ho_fail[u] = True
                continue

            serving[u] = picked
            prev = int(self.prev_serving_sat[u].item())
            if prev >= 0 and prev != picked:
                handover[u] = True

        return serving, handover, ho_fail, incoming_users

    def _make_step(self) -> Dict[str, Any]:
        state = self.ephem.step(self.dt)

        # Scale to normalized units (important for thresholds)
        pos = state.pos * self.pos_scale
        vel = state.vel * self.vel_scale

        user_pos = pos[: self.K]
        sat_pos = pos[self.K :]
        user_vel = vel[: self.K]
        sat_vel = vel[self.K :]

        # Node features: [pos, vel]
        node_x = torch.cat([pos, vel], dim=-1)

        # ---- Build user->sat edges (top-k with radius + fallback) ----
        edge_index_us, edge_ctx = build_user_sat_edges(
            user_pos=user_pos,
            sat_pos=sat_pos,
            visibility_radius=self.visibility_radius,
            user_offset=0,
            sat_offset=self.K,
            topk=self.topk,
            fallback_to_topk=True,
        )

        # ---- user->sat edge descriptors ----
        if edge_index_us.numel() == 0:
            edge_z_us = torch.zeros((0, 6), device=self.device)
            edge_type_us = torch.zeros((0,), dtype=torch.long, device=self.device)

            serving = torch.full((self.K,), -1, device=self.device, dtype=torch.long)
            handover = torch.zeros((self.K,), device=self.device, dtype=torch.bool)
            ho_fail = torch.ones((self.K,), device=self.device, dtype=torch.bool)
            incoming_users = torch.zeros((self.S,), device=self.device, dtype=torch.float32)
            rate = torch.zeros((0,), device=self.device)
            sinr = torch.zeros((0,), device=self.device)

        else:
            src_u = edge_index_us[0]                 # [E] global user idx
            dst_s = edge_index_us[1] - self.K        # [E] local sat idx

            sinr = compute_sinr(
                user_pos=user_pos[src_u],
                sat_pos=sat_pos[dst_s],
                sat_load=self.sat_load[dst_s],
                base_sinr=self.base_sinr,
                dist_scale=self.dist_scale,
            )
            rate = sinr_to_rate(sinr)

            # Placeholder signaling cost (you can replace with beam-level mapping later)
            ho_cost = torch.full_like(rate, self.cost_inter_sat)

            # Ensure descriptor context has the expected keys
            edge_ctx["src_u"] = edge_ctx.get("src_u", src_u)
            edge_ctx["dst_s"] = edge_ctx.get("dst_s", dst_s)
            edge_ctx["K_users"] = self.K
            edge_ctx["S_sats"] = self.S

            edge_z_us = compute_edge_descriptors(
                edge_ctx=edge_ctx,
                rate=rate,
                ho_cost=ho_cost,
                sat_load=self.sat_load,
                cox_cfg=self.cox_cfg,
                visibility_radius=self.visibility_radius,
                device=self.device,
            )

            edge_type_us = torch.full(
                (edge_z_us.size(0),), int(EdgeType.USER_SAT), device=self.device, dtype=torch.long
            )

            # Greedy multi-user assignment to define ground-truth load dynamics + HO system signals
            serving, handover, ho_fail, incoming_users = self._greedy_assign(
                src_u=src_u,
                dst_s=dst_s,
                rate=rate,
                sinr=sinr,
            )

        # ---- Update load (utilization) ----
        cap = max(1, int(self.sat_capacity_users))
        incoming_util = incoming_users / float(cap)  # utilization increments per sat
        self.sat_load = update_satellite_load(self.sat_load, incoming_util, momentum=self.load_momentum)

        # Update prev serving
        self.prev_serving_sat = serving.detach()

        # ---- Optional sat<->sat proximity edges ----
        edge_index = edge_index_us
        edge_z = edge_z_us
        edge_type = edge_type_us

        if self.include_sat_sat and sat_pos.numel() > 0:
            edge_index_ss_local = build_satellite_interference_edges(sat_pos, radius=self.sat_sat_radius)  # [2,E]
            if edge_index_ss_local.numel() > 0:
                src_s = edge_index_ss_local[0]
                dst_s = edge_index_ss_local[1]
                # offset to global node ids
                edge_index_ss = torch.stack([src_s + self.K, dst_s + self.K], dim=0)

                dist_ss = torch.norm(sat_pos[src_s] - sat_pos[dst_s], dim=-1)
                dist_norm = torch.clamp(dist_ss / (self.sat_sat_radius + 1e-12), 0.0, 1.0)
                proximity = 1.0 - dist_norm

                load_src = self.sat_load[src_s]
                load_feat = torch.tanh(load_src * 2.0)  # keeps values bounded

                edge_z_ss = torch.zeros((edge_index_ss.size(1), 6), device=self.device, dtype=torch.float32)
                # put proximity into "rate" slot to reuse existing dims; keep other physics terms 0
                edge_z_ss[:, 3] = proximity
                edge_z_ss[:, 5] = load_feat

                edge_type_ss = torch.full(
                    (edge_index_ss.size(1),), int(EdgeType.SAT_SAT), device=self.device, dtype=torch.long
                )

                edge_index = torch.cat([edge_index_us, edge_index_ss], dim=1) if edge_index_us.numel() else edge_index_ss
                edge_z = torch.cat([edge_z_us, edge_z_ss], dim=0) if edge_z_us.numel() else edge_z_ss
                edge_type = torch.cat([edge_type_us, edge_type_ss], dim=0) if edge_type_us.numel() else edge_type_ss

        # ---- Supervision: predict sat_load on satellite nodes; zeros for users ----
        y = torch.zeros((self.N, 1), device=self.device)
        y[self.K :, 0] = self.sat_load

        meta = {
            "K_users": int(self.K),
            "S_sats": int(self.S),
            "sat_capacity_users": int(self.sat_capacity_users),
            "serving_sat": serving,          # [K] local sat id, -1 if failure
            "handover": handover,            # [K] bool
            "ho_fail": ho_fail,              # [K] bool
            "incoming_users": incoming_users, # [S] float (counts)
            "incoming_util": incoming_util,  # [S] float
        }

        return {
            "t": int(self.t),
            "node_x": node_x,
            "edge_index": edge_index,
            "edge_z": edge_z,
            "edge_type": edge_type,
            "y": y,
            "meta": meta,
        }
