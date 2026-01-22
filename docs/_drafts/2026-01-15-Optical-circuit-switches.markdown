Optical circuit switches
(OCSes) dynamically reconfigure its interconnect topology to
improve scale, availability, utilization, modularity, deployment,
security, power, and performance; users can pick a twisted 3D
torus topology if desired. Much cheaper, lower power, and faster
than Infiniband, OCSes and underlying optical components are
<5% of system cost and <3% of system power. 





An **Optical Circuit Switch (OCS)** is a networking device that routes data signals entirely in the form of light (photons) without ever converting them into electrical signals.

In traditional "packet" switches, incoming light is converted to electricity, processed, buffered, and then converted back to light—a process called **OEO (Optical-Electrical-Optical) conversion**. OCS bypasses this, acting like a programmable "patch panel" or a series of tiny mirrors that physically steer light beams from one fiber to another.

---

## 1. How It Works: The "Mirror" Mechanism

The most common technology used in modern OCS is **3D MEMS** (Micro-Electro-Mechanical Systems).

* **The Component:** Thousands of microscopic mirrors are etched onto a silicon wafer.
* **The Action:** Each mirror can tilt on two axes (3D) using electrostatic force.
* **The Path:** When a command is sent to the switch, the mirrors shift their angles to reflect an incoming beam of light from an input port directly into a specific output port.

### Key Technologies

| Technology | Mechanism | Best Use Case |
| --- | --- | --- |
| **MEMS** | Moving micro-mirrors | Large-scale data centers (high port count). |
| **LCoS** | Liquid Crystal on Silicon | Wavelength-selective switching (telecom). |
| **Thermo-Optic** | Changing refractive index via heat | Integrated photonic circuits (small scale). |

---

## 2. Why the Shift to OCS?

As of 2026, OCS has moved from niche telecommunications use into the heart of **AI Data Centers** for several critical reasons:

* **Near-Zero Latency:** Since there is no electrical processing, the "hop" through the switch happens at the speed of light.
* **Protocol Independence:** An OCS doesn't care if you are sending 400G, 800G, or 1.6T signals. It simply reflects the light, making it "future-proof" as speeds increase.
* **Massive Power Savings:** OEO conversion is incredibly power-hungry. By staying in the optical domain, OCS can reduce a data center's switching power consumption by up to **30%–40%**.
* **Reliability:** Fewer electronic components mean fewer heat-related failures and a simpler overall architecture.

---

## 3. OCS vs. Packet Switching

It is important to note that OCS is usually a **complement** to, not a total replacement for, traditional packet switches.

| Feature | Packet Switch (Electrical) | Optical Circuit Switch (OCS) |
| --- | --- | --- |
| **Data Handling** | Examines every "packet" header. | Handles the entire "circuit" of light. |
| **Speed** | Limited by electronic processing. | Limited only by the fiber capacity. |
| **Switching Time** | Nanoseconds (very fast). | Milliseconds (slow—mirrors must move). |
| **Best For** | Bursty, short-lived traffic. | Long-lived, heavy AI/HPC workloads. |

---

## 4. Current Applications (2026)

* **AI Training Clusters:** Google (with its *Apollo* fabric) and other hyperscalers use OCS to interconnect thousands of TPUs or GPUs. When an AI model is being trained, the data paths stay the same for hours; OCS provides a "permanent" high-speed pipe for that duration.
* **Disaggregated Computing:** Allowing a pool of CPUs in one rack to talk to a pool of GPUs in another as if they were on the same motherboard.
* **Disaster Recovery:** Rapidly rerouting massive data trunk lines in telecom backbones if a fiber is cut.

**Would you like me to compare specific OCS vendors like Lumentum or Polatis, or would you prefer a deeper dive into how Google uses OCS in its Jupiter/Apollo networks?**


As of 2026, the answer is **yes, but with a major technical distinction.** NVIDIA has officially moved into the optical switching market, but they are primarily focused on **Co-Packaged Optics (CPO)** rather than the "moving mirror" (MEMS) style of OCS popularized by Google.

Here is how NVIDIA is integrating optical technology into their current lineup:

### 1. The "Photonics" Switch Lineup

In 2025 and 2026, NVIDIA launched a new class of switches that integrate optics directly into the silicon package. Unlike traditional switches that use pluggable transceivers (the little boxes you slide into the front), these switches "fuse" the light-handling components onto the chip itself.

* **Quantum-X800 Photonics (InfiniBand):** Launched in early 2026, this is a liquid-cooled switch designed for AI "factories." It uses silicon photonics to hit speeds of **800Gb/s per port** across 144 ports.
* **Spectrum-X800 Photonics (Ethernet):** Scheduled for mid-to-late 2026, this platform is aimed at multi-tenant AI clouds.

### 2. OCS vs. NVIDIA’s "Photonics" Approach

While both technologies use light, they solve different problems:

| Feature | Google's OCS (MEMS) | NVIDIA's Photonics (CPO) |
| --- | --- | --- |
| **Mechanism** | Physical mirrors tilt to reflect light. | Silicon chips modulate light directly. |
| **Function** | Reconfigures the "topology" of the network. | Shrinks the switch and cuts power by 40%+. |
| **Primary Goal** | Massive scale and flexibility between racks. | Extreme density and low-latency "scale-up." |
| **Data Handling** | Circuit switching (transparent to data). | Packet switching (processes every packet). |

### 3. The "Optical NVLink" (The Future)

NVIDIA’s "Super-Competitive Advantage" is **NVLink**. Historically, NVLink was restricted to a single rack because copper cables can only carry high-speed signals a few meters.

NVIDIA is now using **Silicon Photonics** to extend NVLink. This allows them to connect up to **72 or even 576 GPUs** in a single "logical" pod using fiber optics, effectively turning a whole row of racks into one giant GPU.

### 4. Key Partnerships

NVIDIA isn't building the optics alone. They have deep collaborations with companies that specialize in the physical components of optical switching:

* **Lumentum:** A key supplier of the laser sources and optical engines.
* **TSMC:** Uses their "COUPE" (Compact Universal Photonic Engine) technology to stack the optical layers on top of NVIDIA's chips.

---

**Summary:** NVIDIA uses optical technology to make their **packet switches** more efficient and to extend **NVLink** across racks, but they generally leave the "mirror-based" circuit switching to hyperscalers like Google who build their own custom infrastructure.

**Would you like to know more about the "Blackwell" or "Rubin" architectures that use these optical interconnects?**



In 2026, NVIDIA’s strategy for optical technology has solidified around **Co-Packaged Optics (CPO)** and **Silicon Photonics**. While they don't use the "moving mirror" OCS (Optical Circuit Switch) seen in Google's data centers, they have integrated light directly into their most advanced GPU architectures: **Blackwell** and the newly launched **Rubin**.

---

## 1. Blackwell (2024–2025): The Transition to Optics

In the Blackwell generation, NVIDIA began moving away from traditional pluggable transceivers (the small boxes that convert electricity to light) toward integrated optical solutions for their switches.

* **Quantum-X800 & Spectrum-X800:** These are NVIDIA's first switches to feature **Co-Packaged Optics**. By placing the optical engine directly on the switch's silicon package, NVIDIA reduced power consumption by over **3.5x** compared to older electrical-to-optical methods.
* **NVL72 Rack:** While the internal "Spine" of a single Blackwell rack still uses a massive forest of copper cables (over 2 miles of it!), the connections *between* these racks are where the high-speed optical links take over.

---

## 2. Rubin (2026): The "Optical-First" Architecture

The **Rubin** architecture, launched in early 2026, represents a massive leap in how NVIDIA uses light. It moves beyond just "connecting" chips and begins to treat the entire data center as a single optical computer.

* **NVLink 6.0:** This is the "nervous system" of the Rubin GPU. It delivers **3.6 TB/s** of bandwidth per GPU—double that of Blackwell. To maintain this speed over longer distances (across multiple racks), NVIDIA is using **Silicon Photonics engines** built with TSMC.
* **Micro Ring Modulators:** This is a breakthrough technology NVIDIA uses in Rubin. Instead of bulky lasers, they use tiny "rings" of silicon that can modulate light at 200Gbps per wavelength. This allows them to pack more data into a smaller space with almost no heat.
* **Vera CPU + Rubin GPU Superchip:** These chips are designed to communicate over an "Optical NVLink" bridge, allowing for a pool of up to **576 GPUs** to act as if they are a single processor with unified memory.

---

## 3. Why NVIDIA chose CPO over MEMS-OCS

NVIDIA's goal is **Scale-Up** (making many GPUs act like one), whereas Google's goal with OCS is **Topology Flexibility** (changing which racks talk to which).

| Feature | NVIDIA's Approach (CPO) | Google's Approach (OCS) |
| --- | --- | --- |
| **Location** | Inside the switch/GPU package. | Standalone switch unit in the rack. |
| **Component** | Silicon Photonics (no moving parts). | MEMS (Micro-mirrors that tilt). |
| **Main Benefit** | Extreme density & low power. | Massive reconfigurability. |
| **2026 Status** | Standard in Rubin & Spectrum-X800. | Core of Jupiter/Apollo networks. |

---

### Summary of the "Rubin" Optical Leap (2026)

* **Bandwidth:** 3.6 TB/s per GPU (NVLink 6).
* **Efficiency:** 5x power reduction per port via integrated photonics.
* **Scale:** Allows a 72-GPU "Pod" to scale out to thousands of GPUs with nearly zero latency "jitter" because the signals stay in the optical domain longer.

**Would you like to see a technical breakdown of how the Micro Ring Modulators in the Rubin architecture actually move data, or perhaps a look at the "Vera" CPU that pairs with these GPUs?**