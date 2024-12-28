import React from "react";
import "./App.css";

function App() {
  const [ticker, setTicker] = React.useState("");
  const [startDate, setStartDate] = React.useState("");
  const [endDate, setEndDate] = React.useState("");
  const [sequenceLength, setSequenceLength] = React.useState(30);
  const [units, setUnits] = React.useState(16);
  const [epochsNum, setEpochsNum] = React.useState(5);
  const [plotUrl1, setPlotUrl1] = React.useState(null);
  const [plotUrl2, setPlotUrl2] = React.useState(null);
  const [loading, setLoading] = React.useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPlotUrl1(null);
    setPlotUrl2(null);

    console.log("submitting", {
      ticker,
      start_date: startDate,
      end_date: endDate,
      seq_length: sequenceLength,
      units,
      epochs_num: epochsNum,
    });

    const response = await fetch(`${REACT_APP_API_URL}/train`, {
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
      setPlotUrl1(data.plot_url1);
      setPlotUrl2(data.plot_url2);
    } else {
      alert("Error generating prediction. Please check inputs.");
    }
    setLoading(false);
  };

  return (
    <div className="flex min-h-screen bg-neutral-900">
      {/* Left Panel */}
      <div className="w-1/4 bg-neutral-950 shadow-md px-8 pt-6 pb-8 flex flex-col">
        <h1 className="text-2xl font-bold mb-6 text-neutral-200">
          Stock Prediction Panel
        </h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Ticker */}
          <div>
            <label
              htmlFor="ticker"
              className="block text-neutral-200 text-sm font-bold mb-2">
              Ticker Symbol:
            </label>
            <input
              id="ticker"
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              className="appearance-none border-b-2 border-neutral-200 w-full py-2 px-3 text-neutral-200 leading-tight focus:outline-none focus:border-neutral-500 bg-transparent"
              required
            />
          </div>

          {/* Start Date */}
          <div>
            <label
              htmlFor="start-date"
              className="block text-neutral-200 text-sm font-bold mb-2">
              Start Date:
            </label>
            <input
              id="start-date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="appearance-none border-b-2 border-neutral-200 w-full py-2 px-3 text-neutral-200 leading-tight focus:outline-none focus:border-neutral-500 bg-transparent"
              required
            />
          </div>

          {/* End Date */}
          <div>
            <label
              htmlFor="end-date"
              className="block text-neutral-200 text-sm font-bold mb-2">
              End Date:
            </label>
            <input
              id="end-date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="appearance-none border-b-2 border-neutral-200 w-full py-2 px-3 text-neutral-200 leading-tight focus:outline-none focus:border-neutral-500 bg-transparent"
              required
            />
          </div>

          {/* Sequence Length */}
          <div>
            <label
              htmlFor="sequence-length"
              className="block text-neutral-200 text-sm font-bold mb-2 mt-8">
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
              className="block text-neutral-200 text-sm font-bold mb-2">
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
              className="block text-neutral-200 text-sm font-bold mb-2">
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
              className="bg-neutral-300 hover:bg-neutral-100 mt-8 text-black font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full">
              Submit
            </button>
          </div>
        </form>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col items-center justify-center">
        {loading ? (
          <div className="text-center">
            <div className="loader border-t-4 border-dashed border-blue-500 rounded-full w-16 h-16 animate-spin"></div>
            <h1 className="mt-4 text-neutral-200 font-bold text-md">
              Loading...
            </h1>
          </div>
        ) : plotUrl1 && plotUrl2 ? (
          <div className="text-left">
            <h2 className="text-6xl font-bold mb-4 absolute top-12 left-128 text-neutral-200">
              Results
            </h2>
            <div className="flex justify-center">
              <img
                src={`http://localhost:5000${plotUrl1}?t=${new Date().getTime()}`}
                alt="Stock Price Prediction 1"
                className="mx-2 mb-4"
                style={{ border: "none" }}
              />
              <img
                src={`http://localhost:5000${plotUrl2}?t=${new Date().getTime()}`}
                alt="Stock Price Prediction 2"
                className="mx-2 mb-4"
                style={{ border: "none" }}
              />
            </div>
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
