async function loadDemandByRegion() {
	const response = await fetch("/api/demand-by-region");
	if (!response.ok) {
		throw new Error(`Region demand request failed: ${response.status}`);
	}

	const regions = await response.json();
	const chart = document.getElementById("demandByRegionChart");

	new Chart(chart, {
		type: "bar",
		data: {
			labels: regions.map((item) => item.region),
			datasets: [{
				label: "Units Sold",
				data: regions.map((item) => item.total_demand),
				backgroundColor: "#2f6f73",
				borderColor: "#214e51",
				borderWidth: 1
			}]
		},
		options: {
			responsive: true,
			maintainAspectRatio: false,
			scales: {
				y: {
					beginAtZero: true
				}
			}
		}
	});
}

loadDemandByRegion().catch((error) => {
	console.error(error);
});
