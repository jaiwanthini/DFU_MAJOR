function initializePressureChart(labels, values) {
  const ctx = document.getElementById('pressureChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Pressure (%)',
          data: values,
          backgroundColor: ['#2563eb', '#10b981', '#f59e0b'],
          borderRadius: 12,
          barThickness: 32,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => `${context.parsed.y}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#475569', font: { size: 13 } },
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { color: '#475569', font: { size: 13 }, stepSize: 20 },
          grid: { color: 'rgba(148, 163, 184, 0.16)' },
        },
      },
    },
  });
}
