import React from "react";
import "./App.css";

function App() {
  const [ticker, setTicker] = React.useState("");
  const [startDate, setStartDate] = React.useState("");
  const [endDate, setEndDate] = React.useState("");
  const [sequenceLength, setSequenceLength] = React.useState(30);
  const [units, setUnits] = React.useState(16);
  const [epochsNum, setEpochsNum] = React.useState(5);
  const [plotUrl, setPlotUrl] = React.useState(null);
  // const [prediction, setPrediction] = React.useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    console.log("submitting", {
      ticker,
      start_date: startDate,
      end_date: endDate,
      seq_length: sequenceLength,
      units,
      epochs_num: epochsNum,
    });

    const response = await fetch(`http://localhost:5000/train`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ticker,
        start_date: startDate,
        end_date: endDate,
        seq_length: sequenceLength,
        units,
        epochs_num: epochsNum,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      setPlotUrl(data.plot_url);
      // setPrediction(data.returns.prediction);
    } else {
      alert("Error generating prediction. Please check inputs.");
    }
  };

  return (
    <div className="flex min-h-screen bg-gray-100">
      {/* Left Panel */}
      <div className="w-1/4 bg-white shadow-md px-8 pt-6 pb-8 flex flex-col">
        <h1 className="text-2xl font-bold mb-6">Stock Prediction Panel</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Ticker */}
          <div>
            <label
              htmlFor="ticker"
              className="block text-gray-700 text-sm font-bold mb-2">
              Ticker Symbol:
            </label>
            <input
              id="ticker"
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              required
            />
          </div>

          {/* Start Date */}
          <div>
            <label
              htmlFor="start-date"
              className="block text-gray-700 text-sm font-bold mb-2">
              Start Date:
            </label>
            <input
              id="start-date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              required
            />
          </div>

          {/* End Date */}
          <div>
            <label
              htmlFor="end-date"
              className="block text-gray-700 text-sm font-bold mb-2">
              End Date:
            </label>
            <input
              id="end-date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              required
            />
          </div>

          {/* Sequence Length */}
          <div>
            <label
              htmlFor="sequence-length"
              className="block text-gray-700 text-sm font-bold mb-2">
              Sequence Length: {sequenceLength}
            </label>
            <input
              id="sequence-length"
              type="range"
              min="30"
              max="60"
              step="5"
              value={sequenceLength}
              onChange={(e) => setSequenceLength(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Units */}
          <div>
            <label
              htmlFor="units"
              className="block text-gray-700 text-sm font-bold mb-2">
              Units: {units}
            </label>
            <input
              id="units"
              type="range"
              min="16"
              max="32"
              step="4"
              value={units}
              onChange={(e) => setUnits(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Epochs */}
          <div>
            <label
              htmlFor="epochs-num"
              className="block text-gray-700 text-sm font-bold mb-2">
              Epochs: {epochsNum}
            </label>
            <input
              id="epochs-num"
              type="range"
              min="5"
              max="25"
              step="1"
              value={epochsNum}
              onChange={(e) => setEpochsNum(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Submit Button */}
          <div>
            <button
              type="submit"
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full">
              Submit
            </button>
          </div>
        </form>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col items-center justify-center">
        {plotUrl ? (
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-4">Results</h2>
            <img
              src={`http://localhost:5000${plotUrl}?t=${new Date().getTime()}`}
              alt="Stock Price Prediction"
              className="mx-auto mb-4"
            />
            {/* <p className="text-lg font-semibold">
              Predicted Price: {prediction && prediction.toFixed(2)}
            </p> */}
          </div>
        ) : (
          <h2 className="text-xl font-semibold text-gray-500">
            Please submit the form to see predictions.
          </h2>
        )}
      </div>
    </div>
  );
}

export default App;
