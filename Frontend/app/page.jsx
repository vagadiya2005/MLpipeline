import Link from "next/link"
import { FaBrain } from "react-icons/fa"

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-r from-purple-600 to-pink-400 text-white py-20">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-200">
            Early Detection of Brain Tumors with AI
          </h1>
          <p className="text-xl mb-8">Advanced AI technology for analyzing brain scans to detect tumors early.</p>
          <Link
            href="/scan-upload"
            className="bg-white text-purple-600 px-8 py-3 rounded-full text-lg font-semibold hover:bg-gray-100 transition duration-300"
          >
            Upload Scan
          </Link>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 bg-gray-100">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
          <div className="max-w-md mx-auto">
            <Link
              href="/scan-upload"
              className="bg-white p-6 rounded-lg shadow-lg hover:shadow-xl transition duration-300 transform hover:-translate-y-1 block"
            >
              <div className="text-4xl text-purple-600 mb-4 flex justify-center">
                <FaBrain />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-center">Analyze Brain Scan</h3>
              <p className="text-gray-600 text-center">Upload your MRI or CT scan for AI analysis</p>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12">Our Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                title: "Brain Scan Interpretation",
                description: "AI-powered analysis of MRI and CT scans",
                link: "/scan-upload",
              },
              { title: "Early Detection", description: "Identify potential tumors at early stages", link: "/about" },
              {
                title: "Secure Data Handling",
                description: "Your medical data is protected with advanced encryption",
                link: "/privacy",
              },
            ].map((feature, index) => (
              <Link
                key={index}
                href={feature.link}
                className="bg-white p-6 rounded-lg shadow-md hover:shadow-xl transition duration-300 transform hover:-translate-y-1"
              >
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

