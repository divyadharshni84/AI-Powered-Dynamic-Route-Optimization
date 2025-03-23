import React from "react";
import { Bar } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const Metrics = () => {
  const data = {
    labels: ["Fuel Efficiency", "CO₂ Reduction", "Delivery Time"],
    datasets: [
      {
        label: "Metrics",
        data: [75, 50, 30],  // Example data
        backgroundColor: ["#4caf50", "#2196f3", "#ff9800"],
      },
    ],
  };

  return (
    <div className="metrics">
      <h2>Real-Time Metrics</h2>
      <Bar data={data} />
    </div>
  );
};

export default Metrics;
