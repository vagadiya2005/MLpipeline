import Link from "next/link"

const Header = () => {
  return (
    <header className="sticky top-0 z-50 bg-white shadow-md">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <Link
          href="/"
          className="text-2xl font-bold text-purple-600 bg-gradient-to-r from-purple-600 to-pink-400 bg-clip-text text-transparent"
        >
          BrainScan AI
        </Link>
        <nav>
          <ul className="flex space-x-6">
            <li>
              <Link href="/" className="text-gray-600 hover:text-purple-600 transition duration-300">
                Home
              </Link>
            </li>
            <li>
              <Link href="/scan-upload" className="text-gray-600 hover:text-purple-600 transition duration-300">
                Upload Scan
              </Link>
            </li>
            <li>
              <Link href="/about" className="text-gray-600 hover:text-purple-600 transition duration-300">
                About
              </Link>
            </li>
            <li>
              <Link href="/contact" className="text-gray-600 hover:text-purple-600 transition duration-300">
                Contact
              </Link>
            </li>
          </ul>
        </nav>
        <div className="flex space-x-4">
          <Link
            href="/login"
            className="px-4 py-2 text-purple-600 border border-purple-600 rounded hover:bg-purple-600 hover:text-white transition duration-300"
          >
            Login
          </Link>
          <Link
            href="/register"
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition duration-300"
          >
            Register
          </Link>
        </div>
      </div>
    </header>
  )
}

export default Header

