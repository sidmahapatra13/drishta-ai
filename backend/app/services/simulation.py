"""District deployment simulation - PS requirement 5.

STUB analytical queueing model. Replaced on day 3 by the Simulink/SimEvents
model, which exports scenario results as JSON that this module loads instead.

Degradation path: if SimEvents is not covered by the team's licence, the same
discrete-event simulation is written as a MATLAB script with identical
parameters and outputs. The dashboard consumes JSON either way. Whichever was
used must be stated honestly on the slide - the numbers on the dashboard must
never be typed by hand.
"""

from dataclasses import asdict, dataclass

WORKING_DAYS = 250

#: Utilization above this queues badly under variable arrivals, so it is the
#: service-level threshold rather than full saturation at 1.0.
SERVICE_LEVEL = 0.85


@dataclass
class Scenario:
    name: str
    patients_per_year: int = 100_000
    health_centres: int = 25
    doctors: int = 8
    bandwidth_mbps: float = 2.0
    images_per_patient: int = 2
    image_size_mb: float = 3.0
    quality_check_sec: float = 1.5
    ai_inference_sec: float = 4.0
    doctor_review_sec: float = 30.0
    referral_rate: float = 0.22
    hours_per_day: float = 8.0
    screening_stations: int = 25
    capture_minutes_per_patient: float = 5.0
    #: Ophthalmologists do not read fundus images all day. This is the hours
    #: per day they actually spend on screening review alongside clinic duties,
    #: and it is the parameter that decides whether the programme is feasible.
    doctor_review_hours_per_day: float = 2.0


def run(s: Scenario) -> dict:
    """M/M/c-style capacity estimate. Indicative only until Simulink replaces it."""
    patients_per_day = s.patients_per_year / WORKING_DAYS
    images_per_day = patients_per_day * s.images_per_patient
    seconds_per_day = s.hours_per_day * 3600

    # Transfer capacity: how many images the link can carry in a working day.
    transfer_sec_per_image = (s.image_size_mb * 8) / s.bandwidth_mbps
    network_capacity = seconds_per_day / transfer_sec_per_image

    # AI capacity assumes one inference worker per health centre.
    ai_sec_per_image = s.quality_check_sec + s.ai_inference_sec
    ai_capacity = (seconds_per_day / ai_sec_per_image) * s.health_centres

    # Physical capture capacity - often the real constraint, since a health
    # worker can only photograph so many patients in a day.
    station_capacity = (s.hours_per_day * 60 / s.capture_minutes_per_patient) * s.screening_stations

    # Doctors review only the referable fraction, and only for the part of the
    # day they are actually available for screening review.
    reviews_per_day = patients_per_day * s.referral_rate
    doctor_seconds = s.doctor_review_hours_per_day * 3600
    doctor_capacity = (doctor_seconds / s.doctor_review_sec) * s.doctors

    utilization = {
        "capture": patients_per_day / station_capacity,
        "network": images_per_day / network_capacity,
        "ai": images_per_day / ai_capacity,
        "doctor": reviews_per_day / doctor_capacity,
    }
    bottleneck = max(utilization, key=utilization.get)

    # Queue length grows sharply as utilization approaches 1 (rho / (1 - rho)).
    rho = min(utilization["doctor"], 0.999)
    queue_length = rho / (1 - rho) if rho < 1 else float("inf")

    # The value proposition, quantified: without AI triage every screened
    # patient needs a specialist read, not just the referable fraction.
    doctor_util_without_ai = patients_per_day / doctor_capacity

    return {
        "scenario": s.name,
        "config": asdict(s),
        "patients_per_day": round(patients_per_day, 1),
        "images_per_day": round(images_per_day, 1),
        "referrals_per_day": round(reviews_per_day, 1),
        "capacity": {
            "capture_patients_per_day": round(station_capacity, 1),
            "network_images_per_day": round(network_capacity, 1),
            "ai_images_per_day": round(ai_capacity, 1),
            "doctor_reviews_per_day": round(doctor_capacity, 1),
        },
        "utilization": {k: round(v, 3) for k, v in utilization.items()},
        "bottleneck": bottleneck,
        "avg_doctor_queue": round(queue_length, 1) if queue_length != float("inf") else None,
        "doctor_utilization_without_ai": round(doctor_util_without_ai, 3),
        "ai_triage_reduction": round(1 - s.referral_rate, 3),
        # A resource above 85% utilization queues badly even though it has not
        # formally saturated, so that - not 100% - is the service-level test.
        "meets_demand": all(v < SERVICE_LEVEL for v in utilization.values()),
        "source": "analytical-stub",
    }


DEFAULT_SCENARIOS = [
    Scenario("A: 4 doctors, 1 Mbps, 10 centres",
             doctors=4, bandwidth_mbps=1.0, health_centres=10, screening_stations=10),
    Scenario("B: 8 doctors, 2 Mbps, 25 centres",
             doctors=8, bandwidth_mbps=2.0, health_centres=25, screening_stations=25),
    Scenario("C: 12 doctors, 5 Mbps, 40 centres",
             doctors=12, bandwidth_mbps=5.0, health_centres=40, screening_stations=40),
]
