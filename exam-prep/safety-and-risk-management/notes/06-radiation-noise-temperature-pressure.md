# Lecture 6 — Ionizing Radiation, Noise, Temperature & Pressure Deviations

⏱ Study time: 60 min | 🔥 Exam weight: **MEDIUM-HIGH** (numbers-heavy; great source of easy marks)

---

## 1. Ionizing Radiation

**Definition:** energy that can **detach electrons from atoms, creating ions** — this damages living cells and DNA.

### ⭐ The 4 types (memorise the table — classic exam table)
| Type | Composition | Penetration | Stopped by | Primary hazard |
|---|---|---|---|---|
| **Alpha (α)** | Helium nucleus (2 protons + 2 neutrons) | **Very low** | Paper, outer layer of skin | **Internal exposure** — inhalation/ingestion |
| **Beta (β)** | High-energy electron or positron | **Moderate** — penetrates skin | Aluminium, plastics | Skin/eye damage, internal exposure, **Bremsstrahlung X-rays** |
| **Gamma (γ)** | Energetic photon (EM wave) | **High** — passes right through the body | **Lead or concrete** | Deep tissue damage, cancer, genetic effects |
| **X-ray** | Energetic photon (EM wave) | **High** | Dense shielding (lead) | Deep tissue damage, cancer, genetic effects |

🧠 Memory: **A**lpha = **A** sheet of paper. **B**eta = **B**it of aluminium. **G**amma = **G**reat slab of lead.

### Sources
- **Natural:** **radon gas** (enters buildings through foundation cracks), **cosmic rays**
- **Artificial:** medical/diagnostic (X-ray equipment, fluoroscopy, PET scans), industrial (radioactive materials, industrial radiography), consumer products

### ⭐ Health effects: Deterministic vs Stochastic (very likely exam question)
| | **Deterministic** | **Stochastic** |
|---|---|---|
| Occurs | Only above a **threshold dose** | By **statistical chance** — no threshold |
| Severity | **Increases with dose** above threshold | **Independent of dose**; only the *probability* rises with dose |
| Timing | Usually early/acute | **Long-term**, latency of many years |
| Examples | Skin reddening (**erythema**), **Acute Radiation Syndrome (ARS)**, temporary sterility, cataracts | **Cancer** — leukaemia and solid tumours |

📌 Key implication: because stochastic effects have no threshold, **no radiation dose is considered completely "safe"** — hence ALARA.

### ⭐⭐ ALARA — "As Low As Reasonably Achievable"
The guiding principle of radiation protection: avoid all exposures that give **no direct benefit**, and keep occupational doses **as far below regulatory limits as reasonably achievable**, considering technology, economics and social factors.

**Three basic protective measures — Time, Distance, Shielding:**
| Measure | Principle | Practice |
|---|---|---|
| **Time** | Dose is **directly proportional to time** spent in the field | Work quickly, leave promptly, plan the task, use alarming dosimeters to manage stay time |
| **Distance** | Dose falls as distance rises. For gamma/X-rays the **inverse square law** applies: **doubling the distance reduces the dose rate by a factor of 4** | Remote handling tools (tongs, robotic arms), control booths, exclusion zones |
| **Shielding** | Place an appropriate material between source and worker | Lead/concrete for gamma & X-ray; **low atomic number (Z<14) materials — aluminium, Plexiglas® — for high-energy beta** to avoid producing secondary **Bremsstrahlung** X-rays; water for neutrons; shelter in a building's centre/basement in an emergency |

⚠️ **ALARA ≠ ALARP.** ALARA = radiation, "**A**chievable". ALARP = general risk, "**P**racticable".

### Control measures (layered: engineering → administrative → PPE)

**Engineering controls (4 — a stated exam question in the slides):**
1. **Shielding** — physical barriers of the right material and thickness; NCRP recommends design goals for medical X-ray; lead (leaded glass, sheet lead, lead-lined drywall, lead bricks) or dense concrete; portable shielding for specific tasks; **design must be done by a qualified expert (health physicist)**
2. **Interlock systems** — automatically shut off/reduce radiation emission when an access point (door to a controlled area) is opened; use door pressure sensors, motion detectors, key control for high-energy sources
3. **Radioactive material containment** — containment integrated into shielding (gamma cameras, industrial radiography devices) to prevent release
4. *(also)* **Ventilation / fume containment** for airborne radionuclides

**Administrative controls (4):**
1. **Posting provisions (signs)** — "Caution Radiation Area" (**>5 mrem/hr**), "Caution High Radiation Area" (**>100 mrem/hr**), "Caution Airborne Radioactivity Area", "Caution Radioactive Material" — per **OSHA 29 CFR 1910.1096(e)**
2. **Warning systems** — audible/visible alarms when radiation is being emitted
3. **Written operating procedures** — normal operations and emergencies (spills, leaks, evacuation)
4. **Work planning** — direct application of ALARA time and distance

**PPE (last resort):** PPE mainly prevents **contamination**, not external dose (unless it contains shielding).
1. **Alpha:** gloves, lab coats, safety glasses; **respirators critical** against airborne alpha (internal exposure)
2. **Beta:** safety goggles, gloves, lab coats
3. **X-ray/gamma:** **lead aprons/vests, lead thyroid collars, lead gloves, leaded eyewear**
4. **Respirators** — last choice for internal exposure to airborne radionuclides; properly selected, worn, maintained
5. **Alarming dosimeters** — personal detectors to manage stay time and track accumulated dose

---

## 2. Noise Pollution ⭐ (numbers = easy marks)

### The numbers (MEMORISE)
| Threshold | Value | Source |
|---|---|---|
| Definition of noise pollution | **> 65 dB** | WHO |
| Harmful noise | **> 75 dB** | WHO |
| Painful noise | **> 120 dB** | WHO |
| Ideal night-time ambient (restful sleep) | **≤ 30 dB** | WHO |
| Recommended daytime | **< 65 dB** | WHO |
| **Hazardous workplace noise / REL** | **85 dBA over an 8-hour shift** | **NIOSH** |
| **Exchange rate** | **3 dBA increase → halve the exposure duration** | NIOSH |
| Enforceable limits | **PELs (various)** | OSHA |

**Simple field indicators of hazardous noise:** must **raise your voice** to be heard at arm's length ≈ **85 dBA** · must **shout** ≈ **95 dBA or more** · **ringing in ears (tinnitus)** afterwards · sounds seem **dull or flat** after leaving.

### Measurement methods (5)
1. **Sound Level Meter (SLM)** — noise level at a point in time; good for **area** assessment, equipment noise output, **noise maps**
2. **Personal Noise Dosimeter** — worn by the worker; **best for individual personal exposure**; averages over time to compute a **noise dose**
3. **Time Weighted Average (TWA)** — sound level averaged over 8 hours, accounting for levels and durations
4. **Noise Dose** — percentage of the allowable daily exposure; **≥100% means the daily limit has been exceeded**
5. **Exchange rate** — 3 dBA (NIOSH)

### Types of noise
- **Continuous noise** — peaks at intervals of **one second or less** (printing presses, lawn mower, jackhammer)
- **Impact / impulse noise** — peaks separated by **more than one second**; rapid release of compressed gas or collision of solid objects (hammer striking metal, burst of air)

### Health effects
- **Temporary Threshold Shift (TTS)** — temporary hearing decrease
- **Permanent Threshold Shift (PTS)** — irreversible damage to sensory cells (hair cells) in the **cochlea**; **cannot be corrected by surgery or medication**
- **Tinnitus** — ringing/buzzing/humming; can become permanent with repeated exposure
- **Other:** physical & psychological stress, reduced productivity and concentration

### Controls (hierarchy applied to noise)
- **Elimination/Substitution** — remove the source or buy quieter equipment (most effective)
- **Engineering** — isolate the source in an insulated room/enclosure, barriers between source and worker, isolate the worker in a sound-insulated booth; damping, silencers, maintenance
- **Administrative** — process training, **job rotation**, adequate rest breaks, limit access to noisy areas, adjust line speeds
- **PPE** — earplugs/earmuffs; last line of defence; **provide whenever noise exceeds 85 dBA regardless of duration**

### ⭐ Hearing Loss Prevention Programme (required if noise ≥ 85 dBA) — 4 components
1. **Noise monitoring** — SLMs for areas, dosimeters for individuals
2. **Audiometric testing** — **baseline test within 30 days** of initial exposure, ideally after a **12-hour quiet period**; **annual tests** performed as close to the **end of the shift** as possible
3. **Worker training and communication** — inform workers of their monitoring results, daily noise levels, in **plain language**; share noise maps, post signs
4. **Provision of hearing protection** — access + training on proper use and maintenance

---

## 3. Heat Stress

**Definition:** occurs when the body's natural cooling mechanisms are **overwhelmed by excessive heat**. Particularly hazardous when combined with **heavy physical work and high humidity**.

**Causes:** *Environmental* — high outdoor temperature, direct sun, high humidity · *Workplace* — strenuous physical work, heat-emitting processes (furnaces), inadequate/poor ventilation.

**Health effects (in order of severity):** discomfort → **heat cramps** → **heat exhaustion** → **heat stroke** (medical emergency); plus dehydration and salt loss, increased accident risk, and vulnerable individuals.

### ⭐ Measurement: WBGT — Wet Bulb Globe Temperature
- A comprehensive measure of heat stress, valid **in direct sunlight**
- Unlike the **heat index** (temperature + humidity, **shade only**), WBGT accounts for **temperature, humidity, wind speed, sun angle and cloud cover (solar radiation)**
- **Three thermometers:** **wet bulb** (evaporative cooling) + **black globe** (solar radiation) + **dry bulb** (air temperature in shade)
- Used to plan outdoor activities for acclimatised workers and athletes

**Controls:**
- **Engineering** — improve ventilation and air conditioning; screen windows with **external blinds/awnings** (best effect); move or screen off heat-emitting machines; turn off unnecessary heat sources
- **Administrative** — schedule heavy work in cooler parts of the day/year; **work/rest schedules** based on heat indices; job rotation; provide **plain water hygienically and within reasonable distance**; mandatory rest breaks in cool/shaded areas
- **PPE** — cooling vests, reflective clothing, breathable fabrics, hats with neck protection, sunscreen
- **Training** — recognise symptoms, prevention, first aid

---

## 4. Cold Stress

**Definition:** occurs when the body **loses heat faster than it can produce it**, dropping core body temperature.

**Causes:** exposure to extreme cold — outdoor work, refrigerated environments, unheated indoor spaces; plus environmental factors and individual factors (exhaustion, predisposing health conditions).

**Health effects (4):** **Hypothermia · Frostbite · Trench Foot (immersion foot) · Chilblains**

### ⭐ Measurement: Wind Chill Factor
- Indicates how cold people/animals **feel**, based on the rate of heat loss from exposed skin due to the **combined effect of wind and cold**
- Higher wind speed draws heat away faster → perceived temperature much colder than actual air temperature
- Calculated from **air temperature + wind speed**; defined for temperatures **at or below 50 °F** and wind speeds **above 3 mph**

**Controls:**
- **Engineering** — radiant heaters, **wind barriers**, heated shelters for breaks, **insulate metal handles and bars**, maintain heating systems
- **Administrative** — schedule cold jobs for warmer months/warmest part of day; relief or extra workers on long jobs; limit outdoor time on extremely cold days; **provide warm liquids and food**; **buddy system**; **warm-up break schedules** based on temperature and wind speed
- **PPE** — **several layers of loose clothing** (better insulation, unrestricted movement); protect ears, face, hands and feet; waterproof insulated boots, hats (head heat loss), insulated gloves, face protection; carry spare socks/gloves/hats/jackets/blankets
- **Training** — risks, prevention, symptom recognition (self and coworkers), treatment, PPE selection; **avoid touching cold metal with bare skin**

---

## 5. Pressure Deviations

**High-pressure environments:** mainly **pressure vessels and compressed gases**, storing gases/liquids well above atmospheric pressure.

**Main hazards:** **rupture or explosion** of the vessel · **leakage** of contents (flammable, toxic or corrosive) · uncontrolled release of **mechanical energy**.

**Control measures (5):**
1. Proper **training** for personnel
2. **Regular inspections and maintenance**
3. Adherence to pressure limits — **MAWP (Maximum Allowable Working Pressure)**
4. Use of **PPE**
5. Control **environmental conditions** (temperature, corrosion, physical protection of cylinders)

**Low-pressure hazards (for completeness):** high-altitude work/flight and diving → **hypoxia, barotrauma, decompression sickness**.

---

### ✅ Self-test
1. Alpha/beta/gamma: penetration and what stops each. 2. Deterministic vs stochastic — 3 differences + examples. 3. ALARA's three measures; what is the inverse square law effect of doubling distance? 4. NIOSH REL and exchange rate? 5. 85 dBA / 95 dBA field indicators. 6. TTS vs PTS. 7. Four components of a Hearing Loss Prevention Programme. 8. WBGT's three thermometers. 9. Four cold-stress illnesses. 10. What is MAWP?
