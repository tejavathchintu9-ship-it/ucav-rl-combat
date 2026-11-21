"""
CombatSim - visual + physical layer of the combat simulation.

Built on top of the unchanged v5 `JetSim` (aircraft physics, models, clouds).
Adds
  * a FIXED radial safety zone around the home base (the original boundary
    was glued to the enemy, so "leaving the zone" could never really happen),
  * realistic missiles for both sides: motor boost + coast, proportional
    navigation, g-limit, seeker gimbal limit, proximity fuse, miss detection,
  * explosions, smoke trails, ground grid,
  * an independent observer camera: after the UCAV launches, the camera no
    longer follows the UCAV - it frames the missile, the enemy and the kill
    while the UCAV breaks away and returns to the zone,
  * HUD, target brackets and a top-down radar.
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

from panda3d.core import (
    CardMaker, Filename, LineSegs, NodePath, Point2, Quat, TextNode,
    TransparencyAttrib, Vec3, Vec4, Material,
)
from direct.gui.OnscreenText import OnscreenText

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "simulator"))
import updated_chase_jet_sim_v5 as base   # noqa: E402
from updated_chase_jet_sim_v5 import JetSim, vec_length_sq  # noqa: E402

# ----------------------------------------------------------------------------
ZONE_RADIUS = 12000.0
HOME = Vec3(0.0, 0.0, 1500.0)

# missile model (shared by both sides) ---------------------------------------
MSL_FREE_FLIGHT = 0.35      # s, unguided after rail release
MSL_BURN = 5.0
MSL_BOOST_ACCEL = 170.0
MSL_MAX_SPEED = 720.0
MSL_COAST_DECEL = 7.0
MSL_MIN_SPEED = 250.0
MSL_A_MAX = 400.0           # m/s^2 lateral (~41 g); tuned so evasion can work
MSL_NAV_GAIN = 4.0
MSL_LIFE = 22.0
MSL_GIMBAL_DEG = 70.0
MSL_GIMBAL_LOSS_TIME = 0.8
MSL_HIT_RADIUS = 25.0
MSL_INDUCED_DRAG = 0.012


def _orient_to(node, direction: Vec3):
    """The missile asset's long axis is local +Z."""
    axis = Vec3(0, 0, 1)
    rot = axis.cross(direction)
    al = max(-1.0, min(1.0, axis.dot(direction)))
    q = Quat()
    if vec_length_sq(rot) < 1e-8:
        if al < 0:
            q.setFromAxisAngle(180.0, Vec3(1, 0, 0))
    else:
        rot.normalize()
        q.setFromAxisAngle(math.degrees(math.acos(al)), rot)
    node.setQuat(q)


class Trail:
    def __init__(self, parent, color, thickness=2.0, maxlen=140, every=2):
        self.parent, self.color, self.th = parent, color, thickness
        self.pts = deque(maxlen=maxlen)
        self.every, self.n = every, 0
        self.np = None

    def add(self, p):
        self.n += 1
        if self.n % self.every == 0:
            self.pts.append(Vec3(p))

    def redraw(self):
        if self.np is not None:
            self.np.removeNode()
            self.np = None
        if len(self.pts) < 2:
            return
        ls = LineSegs()
        ls.setThickness(self.th)
        k = len(self.pts)
        for i, p in enumerate(self.pts):
            a = (i + 1) / k
            ls.setColor(self.color[0], self.color[1], self.color[2], self.color[3] * a)
            (ls.moveTo if i == 0 else ls.drawTo)(p)
        self.np = self.parent.attachNewNode(ls.create())
        self.np.setLightOff()
        self.np.setTransparency(TransparencyAttrib.MAlpha)

    def clear(self):
        self.pts.clear()
        if self.np is not None:
            self.np.removeNode()
            self.np = None


class CombatSim(JetSim):
    # ------------------------------------------------------------------ init
    def __init__(self, zone_radius=ZONE_RADIUS, home=HOME):
        self.zone_radius = float(zone_radius)
        self.home = Vec3(home)
        self._pre_init = True
        super().__init__()
        self._pre_init = False
        self.taskMgr.remove("update_task")
        self.visual = self.win is not None

        self.missiles = []          # active missile dicts
        self.events = []            # consumed by the environment
        self.explosions = []
        self.enemy_destroyed = False
        self.ucav_destroyed = False
        self.sim_time = 0.0

        # camera director state
        self.cam_mode = "auto"      # auto | chase | cockpit
        self._cam_pos = Vec3(0, -50, 1500)
        self._cam_look = Vec3(0, 0, 1500)
        self._observer_angle = 0.6
        self._was_observer = False
        self._observer_hold = 0.0
        self._cam_initialised = False

        self.hud_left = ""
        self.hud_banner = ""
        self.hud_banner_color = (1, 1, 1, 1)
        self.radar_extra = []

        if self.visual:
            self._build_visual_layer()

    # --- neutralise original behaviours that conflict with the new design ----
    def create_hud(self):
        self.hud_text = None

    def load_boundary_sphere(self):
        self.enemy.boundary_sphere = None

    def setup_controls(self):
        self.accept("escape", sys.exit)
        self.accept("c", self._cycle_cam)

    def _cycle_cam(self):
        order = ["auto", "chase", "cockpit"]
        self.cam_mode = order[(order.index(self.cam_mode) + 1) % 3]

    def update_hud(self):
        pass

    # ----------------------------------------------------------- visual layer
    def _build_visual_layer(self):
        self.camLens.setNearFar(4.0, 70000.0)
        self.camLens.setFov(68)
        self.setBackgroundColor(0.50, 0.74, 0.93, 1)
        self.fx_root = self.render.attachNewNode("fx")
        self._build_ground_and_zone()
        self.trail_enemy = Trail(self.fx_root, (1.0, 0.45, 0.2, 0.55), 2.0, 160, 3)
        self.trail_ucav = Trail(self.fx_root, (0.2, 0.9, 0.3, 0.55), 2.0, 160, 3)
        self.trail_missile = {}

        a2d = self.aspect2d
        self.t_left = OnscreenText(text="", pos=(-1.72, 0.94), scale=0.036, fg=(1, 1, 0.55, 1),
                                   bg=(0, 0, 0, 0.45), align=TextNode.ALeft, mayChange=True,
                                   parent=a2d)
        self.t_banner = OnscreenText(text="", pos=(0.75, 0.92), scale=0.05, fg=(1, 1, 1, 1),
                                     shadow=(0, 0, 0, 1), align=TextNode.ACenter, mayChange=True,
                                     parent=a2d)
        self.t_big = OnscreenText(text="", pos=(0, 0.1), scale=0.14, fg=(1, 1, 1, 1),
                                  shadow=(0, 0, 0, 1), align=TextNode.ACenter, mayChange=True,
                                  parent=a2d)
        self.brackets = {}
        self._build_radar()

    def _build_ground_and_zone(self):
        # ground grid -> gives speed / altitude cues
        ls = LineSegs()
        ls.setThickness(1.0)
        ls.setColor(0.30, 0.45, 0.28, 1)
        ext, step = 40000, 1000
        for v in range(-ext, ext + 1, step):
            ls.moveTo(v, -ext, 0); ls.drawTo(v, ext, 0)
            ls.moveTo(-ext, v, 0); ls.drawTo(ext, v, 0)
        g = self.render.attachNewNode(ls.create())
        g.setLightOff()
        cm = CardMaker("ground")
        cm.setFrame(-60000, 60000, -60000, 60000)
        gp = self.render.attachNewNode(cm.generate())
        gp.setP(-90); gp.setPos(0, 0, -2)
        gp.setColor(0.36, 0.52, 0.32, 1); gp.setLightOff()
        # zone: rings (equator + two meridians + ground circle) in orange
        zl = LineSegs(); zl.setThickness(2.0); zl.setColor(1.0, 0.55, 0.1, 0.9)
        R, N = self.zone_radius, 128
        def ring(fn):
            for i in range(N + 1):
                a = 2 * math.pi * i / N
                p = fn(a)
                (zl.moveTo if i == 0 else zl.drawTo)(p)
        cz = self.home.z
        ring(lambda a: Vec3(R * math.cos(a), R * math.sin(a), 0))
        ring(lambda a: Vec3(R * math.cos(a), R * math.sin(a), cz))
        ring(lambda a: Vec3(R * math.cos(a), 0, cz + R * math.sin(a) * 0.35))
        ring(lambda a: Vec3(0, R * math.cos(a), cz + R * math.sin(a) * 0.35))
        zn = self.render.attachNewNode(zl.create())
        zn.setLightOff(); zn.setTransparency(TransparencyAttrib.MAlpha)
        # home beacon
        hb = LineSegs(); hb.setThickness(3); hb.setColor(0.2, 1, 0.4, 1)
        hb.moveTo(0, 0, 0); hb.drawTo(0, 0, 2500)
        hbn = self.render.attachNewNode(hb.create()); hbn.setLightOff()

    def _build_radar(self):
        self.radar = self.aspect2d.attachNewNode("radar")
        self.radar.setPos(1.40, 0, -0.60)
        self.radar.setScale(0.36)
        cm = CardMaker("rbg"); cm.setFrame(-1.05, 1.05, -1.05, 1.05)
        bg = self.radar.attachNewNode(cm.generate())
        bg.setColor(0, 0.08, 0, 0.55); bg.setTransparency(TransparencyAttrib.MAlpha)
        ls = LineSegs(); ls.setThickness(1.5); ls.setColor(1, 0.55, 0.1, 1)
        for rr, c in ((1.0, (1, 0.55, 0.1, 1)), (0.66, (0.2, 0.5, 0.2, 1)), (0.33, (0.2, 0.5, 0.2, 1))):
            ls.setColor(*c)
            for i in range(65):
                a = 2 * math.pi * i / 64
                (ls.moveTo if i == 0 else ls.drawTo)(rr * math.cos(a), 0, rr * math.sin(a))
        ls.setColor(0.2, 0.5, 0.2, 1)
        ls.moveTo(-1, 0, 0); ls.drawTo(1, 0, 0); ls.moveTo(0, 0, -1); ls.drawTo(0, 0, 1)
        self.radar.attachNewNode(ls.create())
        self.radar_dots = {}
        for key, col, sz in (("ucav", (0.2, 1, 0.3, 1), 0.07), ("enemy", (1, 0.2, 0.2, 1), 0.07),
                             ("m_ucav", (1, 1, 0.2, 1), 0.045), ("m_enemy", (1, 0.3, 1, 1), 0.045)):
            c = CardMaker(key); c.setFrame(-sz, sz, -sz, sz)
            n = self.radar.attachNewNode(c.generate()); n.setColor(*col); n.hide()
            self.radar_dots[key] = n
        self.radar_heading = None
        self.radar_label = OnscreenText(text="RADAR  (zone R=%.0f km)" % (self.zone_radius / 1000),
                                        pos=(1.40, -0.20), scale=0.032, fg=(0.7, 1, 0.7, 1),
                                        parent=self.aspect2d, mayChange=False)

    # -------------------------------------------------------------- scene reset
    def reset_scene(self):
        for m in self.missiles:
            self._kill_node(m)
        self.missiles = []
        self.events = []
        for e in self.explosions:
            e["node"].removeNode()
        self.explosions = []
        self.enemy_destroyed = self.ucav_destroyed = False
        self.player.root.show(); self.enemy.root.show()
        self.missile_fired = False
        if self.missile is not None:
            self.missile.show()
        self.sim_time = 0.0
        self._cam_initialised = False
        self._observer_hold = 0.0
        if self.visual:
            for t in (self.trail_enemy, self.trail_ucav):
                t.clear()
            for t in list(self.trail_missile.values()):
                t.clear()
            self.trail_missile = {}
            self.t_big.setText("")

    def _kill_node(self, m):
        fl = m.get("flame")
        if fl is not None:
            try:
                fl.removeNode()
            except Exception:
                pass
            m["flame"] = None
        n = m.get("node")
        if n is not None:
            try:
                n.removeNode()
            except Exception:
                pass
            m["node"] = None

    # ------------------------------------------------------------------ zone
    def zone_distance(self, pos: Vec3) -> float:
        return (pos - self.home).length()

    def check_boundary(self):          # superseded by environment logic
        pass

    # ------------------------------------------------------------- missiles
    def launch_missile(self, owner: str):
        """owner: 'ucav' or 'enemy'. Returns the missile dict."""
        shooter = self.player if owner == "ucav" else self.enemy
        target = self.enemy if owner == "ucav" else self.player
        shooter.sync_graphics()
        fwd = shooter.velocity.normalized() if vec_length_sq(shooter.velocity) > 1e-6 else shooter.get_forward_vector()
        if owner == "ucav" and self.missile is not None and not self.missile.isHidden():
            pos = Vec3(self.missile.getPos(self.render))
        else:
            pos = shooter.position - shooter.get_up_vector() * 2.0
        m = {
            "owner": owner, "target": target, "pos": Vec3(pos),
            "vel": fwd * (shooter.current_speed + 35.0), "speed": shooter.current_speed + 35.0,
            "age": 0.0, "lost": 0.0, "node": None, "min_range": 1e9, "id": id(object()),
            "prev_target_pos": Vec3(target.position),
        }
        if self.visual:
            node = self.loader.loadModel(Filename.fromOsSpecific(base.asset_path(base.MISSILE_MODEL_PATH)))
            node.reparentTo(self.render)
            node.setPos(pos)
            node.setScale(self.player.visual_root.getScale(self.render).x * base.MISSILE_RELATIVE_SCALE)
            try:
                for ts in node.findAllTextureStages():
                    node.setTexture(ts, None)
                mat = Material(); mat.setDiffuse((0.9, 0.9, 0.9, 1)); mat.setAmbient((0.9, 0.9, 0.9, 1))
                node.setMaterial(mat, 1); node.setTwoSided(True)
            except Exception:
                pass
            if owner == "enemy":
                node.setColorScale(1.0, 0.55, 0.55, 1)
            fl = self.loader.loadModel("models/misc/sphere")
            fl.reparentTo(self.fx_root); fl.setLightOff(); fl.setScale(6.0)
            fl.setTransparency(TransparencyAttrib.MAlpha)
            fl.setColor(1.0, 0.65, 0.2, 0.9) if owner == "ucav" else fl.setColor(1.0, 0.3, 0.9, 0.9)
            m_flame = fl
            _orient_to(node, fwd)
            m["node"] = node
            m["flame"] = m_flame
            col = (1, 0.95, 0.7, 0.85) if owner == "ucav" else (1, 0.5, 0.9, 0.85)
            self.trail_missile[m["id"]] = Trail(self.fx_root, col, 3.0, 220, 1)
        if owner == "ucav":
            if self.missile is not None:
                self.missile.hide()
            self.missile_fired = True
            self._observer_hold = 0.0
        self.missiles.append(m)
        return m

    def own_missile(self):
        for m in self.missiles:
            if m["owner"] == "ucav":
                return m
        return None

    def inbound_enemy_missile(self):
        for m in self.missiles:
            if m["owner"] == "enemy":
                return m
        return None

    def update_missiles(self, dt):
        kept = []
        for m in self.missiles:
            tgt = m["target"]
            m["age"] += dt
            r = tgt.position - m["pos"]
            R = r.length()
            m["min_range"] = min(m["min_range"], R)
            vhat = m["vel"].normalized() if vec_length_sq(m["vel"]) > 1e-9 else Vec3(0, 1, 0)

            # -- guidance ------------------------------------------------
            a_vec = Vec3(0, 0, 0)
            if m["age"] > MSL_FREE_FLIGHT and R > 1e-3 and m["lost"] < MSL_GIMBAL_LOSS_TIME:
                gimbal = math.degrees(math.acos(max(-1.0, min(1.0, (r / R).dot(vhat)))))
                if gimbal > MSL_GIMBAL_DEG:
                    m["lost"] += dt
                else:
                    vrel = tgt.velocity - m["vel"]
                    Vc = -r.dot(vrel) / R
                    omega = r.cross(vrel) / (R * R)
                    a_vec = omega.cross(vhat) * (MSL_NAV_GAIN * max(Vc, 50.0))
                    a_vec = a_vec - vhat * a_vec.dot(vhat)
                    mag = a_vec.length()
                    if mag > MSL_A_MAX:
                        a_vec *= MSL_A_MAX / mag
            # -- propulsion ---------------------------------------------
            sp = m["speed"]
            if m["age"] < MSL_BURN:
                sp = min(MSL_MAX_SPEED, sp + MSL_BOOST_ACCEL * dt)
            else:
                sp = max(MSL_MIN_SPEED, sp - (MSL_COAST_DECEL + MSL_INDUCED_DRAG * a_vec.length()) * dt)
            new_dir = vhat + (a_vec * dt) / max(sp, 1.0)
            new_dir.normalize()
            m["speed"] = sp
            m["vel"] = new_dir * sp
            prev = Vec3(m["pos"])
            m["pos"] = prev + m["vel"] * dt

            # -- proximity fuse via swept closest approach ----------------
            r0 = prev - m["prev_target_pos"]
            r1 = m["pos"] - tgt.position
            seg = r1 - r0
            L2 = seg.dot(seg)
            u = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, -r0.dot(seg) / L2))
            dmin = (r0 + seg * u).length()
            m["prev_target_pos"] = Vec3(tgt.position)

            outcome = None
            if dmin <= MSL_HIT_RADIUS:
                outcome = ("hit", "proximity_fuse")
            elif m["age"] > MSL_LIFE:
                outcome = ("miss", "lifetime")
            elif m["lost"] >= MSL_GIMBAL_LOSS_TIME:
                outcome = ("miss", "seeker_lost_lock")
            elif m["pos"].z < 0.0:
                outcome = ("miss", "ground")
            elif m["age"] > MSL_BURN and R > 700.0 and r.dot(m["vel"]) < 0.0 and m["min_range"] < R - 80.0:
                outcome = ("miss", "flew_past")
            elif sp <= MSL_MIN_SPEED + 1 and m["age"] > MSL_BURN + 6.0 and R > 3000:
                outcome = ("miss", "out_of_energy")

            if self.visual and m["node"] is not None:
                m["node"].setPos(m["pos"])
                _orient_to(m["node"], new_dir)
                fl = m.get("flame")
                if fl is not None:
                    if m["age"] < MSL_BURN:
                        fl.setPos(m["pos"] - new_dir * 7.0)
                        fl.setScale(5.0 + 2.0 * math.sin(m["age"] * 60.0))
                    else:
                        fl.hide()
                self.trail_missile[m["id"]].add(m["pos"])

            if outcome is None:
                kept.append(m)
                continue
            kind, reason = outcome
            pos = Vec3(m["pos"])
            if kind == "hit":
                pos = Vec3(tgt.position)
                if m["owner"] == "ucav":
                    self.enemy_destroyed = True
                else:
                    self.ucav_destroyed = True
                tgt.crashed = True
                self.spawn_explosion(pos, big=True)
                if self.visual:
                    tgt.root.hide()
            else:
                self.spawn_explosion(pos, big=False)
            self.events.append({"type": kind, "owner": m["owner"], "reason": reason,
                                "pos": pos, "miss_distance": m["min_range"]})
            if m["owner"] == "ucav":
                self._last_event_pos = Vec3(pos)
            if self.visual:
                self._observer_hold = 4.5 if m["owner"] == "ucav" else self._observer_hold
            self._kill_node(m)
        self.missiles = kept

    # ------------------------------------------------------------- effects
    def spawn_explosion(self, pos: Vec3, big=True):
        if not self.visual:
            return
        node = self.loader.loadModel("models/misc/sphere")
        node.reparentTo(self.fx_root)
        node.setPos(pos)
        node.setLightOff()
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setScale(1.0)
        node.setColor(1.0, 0.7, 0.2, 0.95)
        self.explosions.append({"node": node, "t": 0.0, "big": big})

    def _update_effects(self, dt):
        for e in list(self.explosions):
            e["t"] += dt
            t = e["t"]
            life = 3.2 if e["big"] else 1.2
            if t > life:
                e["node"].removeNode()
                self.explosions.remove(e)
                continue
            grow = min(1.0, t / 0.45)
            size = (110.0 if e["big"] else 35.0) * (1 - (1 - grow) ** 2) * (1 + 0.15 * t)
            fade = max(0.0, 1.0 - t / life)
            e["node"].setScale(max(size, 1.0))
            g = 0.75 * (1 - min(1.0, t / 1.4)) + 0.1
            e["node"].setColor(1.0, g, 0.1 + 0.1 * (1 - fade), 0.95 * fade)

    # ------------------------------------------------------------- camera
    def _smooth(self, cur: Vec3, tgt: Vec3, rate: float, dt: float) -> Vec3:
        a = 1.0 - math.exp(-rate * dt)
        return cur + (tgt - cur) * a

    def update_camera(self, dt=0.02):
        if not self.visual:
            return
        p, e = self.player, self.enemy
        my = self.own_missile()
        observer = (my is not None or self._observer_hold > 0.0) and self.cam_mode == "auto"
        if observer and my is None:
            self._observer_hold -= dt

        if observer != self._was_observer:
            self._cam_initialised = False           # hard cut when the view changes
            self._was_observer = observer
        if observer:
            # Free observer camera: it does not follow the UCAV at all.
            Z = Vec3(0, 0, 1)
            M = Vec3(my["pos"]) if my is not None else Vec3(self._last_event_pos)
            E = Vec3(e.position) if not self.enemy_destroyed else Vec3(self._last_event_pos)
            span = (E - M).length()
            dirME = (E - M) / span if span > 1.0 else Vec3(0, 1, 0)
            side = dirME.cross(Z)
            side = side / side.length() if side.length() > 1e-6 else Vec3(1, 0, 0)
            if my is not None:
                # Fit-to-frame: choose the distance from the lens FOV so that BOTH the
                # missile and the enemy are inside the picture, seen from the side so the
                # closing geometry is readable.  The UCAV is not a framing target.
                half_h = math.tan(math.radians(self.camLens.getFov()[0]) * 0.5)
                half_v = half_h * 9.0 / 16.0
                mid = (M + E) * 0.5
                vspan = abs(E.z - M.z)
                hspan = math.hypot(E.x - M.x, E.y - M.y)
                d = max(260.0, 1.30 * max(hspan / (2 * half_h), vspan / (2 * half_v)) + 0.45 * hspan)
                self._observer_angle = max(-0.6, min(0.6, self._observer_angle))
                self._observer_angle += dt * 0.05
                ang = self._observer_angle
                horiz = dirME * 1.0
                horiz = Vec3(horiz.x, horiz.y, 0.0)
                hn = horiz.length()
                horiz = horiz / hn if hn > 1e-6 else Vec3(0, 1, 0)
                sidev = horiz.cross(Z)
                tpos = mid + sidev * (d * math.cos(ang)) - horiz * (d * 0.45 * math.sin(ang)) + Z * (0.16 * d + 25.0)
                tlook = mid
                rate = 6.0
            else:
                # impact: slow orbit around the fireball
                self._observer_angle += dt * 0.45
                c, sn = math.cos(self._observer_angle), math.sin(self._observer_angle)
                tpos = E + (side * c + dirME * sn) * 340.0 + Z * 95.0
                tlook = E + Z * 10.0
                rate = 2.5
        elif self.cam_mode == "cockpit":
            fwd = p.get_forward_vector()
            tpos, tlook, rate = p.position + fwd * 6 + Vec3(0, 0, 2), p.position + fwd * 400, 30.0
        else:
            fwd, up = p.get_forward_vector(), p.get_up_vector()
            hf = Vec3(fwd.x, fwd.y, 0.0)
            hf = hf / hf.length() if hf.length() > 1e-6 else Vec3(0, 1, 0)
            tpos = p.position - hf * 55.0 + Vec3(0, 0, 16.0) - fwd * 15.0
            to_e = e.position - p.position
            d = to_e.length()
            look = p.position + fwd * 120.0
            if d < 9000 and not self.enemy_destroyed:
                w = 0.55 if d > 600 else 0.25
                look = look * (1 - w) + e.position * w
            tlook, rate = look, 6.0

        if not self._cam_initialised:
            self._cam_pos, self._cam_look = Vec3(tpos), Vec3(tlook)
            self._cam_initialised = True
        else:
            self._cam_pos = self._smooth(self._cam_pos, tpos, rate, dt)
            self._cam_look = self._smooth(self._cam_look, tlook, rate * 1.4, dt)
        if self._cam_pos.z < 8.0:
            self._cam_pos.z = 8.0
        self.camera.setPos(self._cam_pos)
        self.camera.lookAt(self._cam_look)

    _last_event_pos = Vec3(0, 0, 1500)

    # ------------------------------------------------------- overlay / HUD
    def _bracket(self, key, color):
        if key not in self.brackets:
            ls = LineSegs(); ls.setThickness(2.0); ls.setColor(*color)
            k = 0.6
            for sx in (-1, 1):
                for sz in (-1, 1):
                    ls.moveTo(sx, 0, sz * k); ls.drawTo(sx, 0, sz); ls.drawTo(sx * k, 0, sz)
            n = self.aspect2d.attachNewNode(ls.create())
            lab = OnscreenText(text="", scale=0.034, fg=color, shadow=(0, 0, 0, 1),
                               parent=self.aspect2d, mayChange=True, align=TextNode.ACenter)
            self.brackets[key] = (n, lab)
        return self.brackets[key]

    def _draw_bracket(self, key, world: Vec3, color, label, size_m=22.0, visible=True):
        node, lab = self._bracket(key, color)
        p3 = self.cam.getRelativePoint(self.render, world)
        p2 = Point2()
        if (not visible) or p3.y <= 1.0 or not self.camLens.project(p3, p2):
            node.hide(); lab.hide(); return
        dist = max(p3.length(), 1.0)
        fov_v = math.radians(self.camLens.getFov()[1])
        s = max(0.035, min(0.25, (size_m / dist) / fov_v))
        x, z = p2.x * self.getAspectRatio(), p2.y
        node.show(); node.setPos(x, 0, z); node.setScale(s)
        lab.show(); lab.setPos(x, z - s - 0.045); lab.setText(label)

    def update_overlay(self):
        if not self.visual:
            return
        self.t_left.setText(self.hud_left)
        self.t_banner.setText(self.hud_banner)
        self.t_banner.setFg(self.hud_banner_color)
        e, p = self.enemy, self.player
        R = self.zone_radius
        my = self.own_missile()
        observer = (my is not None or self._observer_hold > 0) and self.cam_mode == "auto"
        # brackets
        dist_e = (e.position - self.camera.getPos(self.render)).length()
        self._draw_bracket("enemy", e.position, (1, 0.25, 0.25, 1),
                           f"ENEMY {(e.position - p.position).length() / 1000:.1f} km",
                           size_m=26.0, visible=not self.enemy_destroyed)
        if my is not None:
            self._draw_bracket("my", my["pos"], (1, 1, 0.3, 1), "MISSILE", size_m=14.0)
        else:
            self._draw_bracket("my", Vec3(0, 0, 0), (1, 1, 0.3, 1), "", visible=False)
        em = self.inbound_enemy_missile()
        if em is not None:
            self._draw_bracket("em", em["pos"], (1, 0.4, 1, 1), "INCOMING", size_m=14.0)
        else:
            self._draw_bracket("em", Vec3(0, 0, 0), (1, 0.4, 1, 1), "", visible=False)
        # radar
        def put(key, pos):
            d = self.radar_dots[key]
            if pos is None:
                d.hide(); return
            rel = pos - self.home
            x, y = rel.x / R, rel.y / R
            L = math.hypot(x, y)
            if L > 1.0:
                x, y = x / L, y / L
            d.show(); d.setPos(x, 0, y)
        put("ucav", None if self.ucav_destroyed else p.position)
        put("enemy", None if self.enemy_destroyed else e.position)
        put("m_ucav", my["pos"] if my else None)
        put("m_enemy", em["pos"] if em else None)
        if self.radar_heading is not None:
            self.radar_heading.removeNode()
        rel = p.position - self.home
        hx, hy = rel.x / R, rel.y / R
        f = p.get_forward_vector()
        ls = LineSegs(); ls.setThickness(2); ls.setColor(0.2, 1, 0.3, 1)
        ls.moveTo(hx, 0, hy); ls.drawTo(hx + f.x * 0.16, 0, hy + f.y * 0.16)
        self.radar_heading = self.radar.attachNewNode(ls.create())

    def set_hud(self, left: str, banner: str, color=(1, 1, 1, 1)):
        self.hud_left, self.hud_banner, self.hud_banner_color = left, banner, color

    def set_big_text(self, text, color=(1, 1, 1, 1)):
        if self.visual:
            self.t_big.setText(text); self.t_big.setFg(color)

    def window_open(self):
        if self.win is None:
            return False
        closed = getattr(self.win, "isClosed", None)
        return not (closed() if closed else False)

    # ---------------------------------------------------------- frame driver
    def render_frame(self, dt=0.02):
        """Update visuals and draw one frame (no physics here)."""
        if not self.visual:
            return True
        if not self.window_open():
            return False
        self.player.sync_graphics(); self.enemy.sync_graphics()
        self.sim_time += dt
        if not self.enemy_destroyed:
            self.trail_enemy.add(self.enemy.position)
        if not self.ucav_destroyed:
            self.trail_ucav.add(self.player.position)
        self._update_effects(dt)
        self.update_camera(dt)
        self.update_overlay()
        # trails are redrawn every other frame to stay cheap
        if int(self.sim_time / dt) % 2 == 0:
            self.trail_enemy.redraw(); self.trail_ucav.redraw()
            for t in self.trail_missile.values():
                t.redraw()
        self.taskMgr.step()
        return True
