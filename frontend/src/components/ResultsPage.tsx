import React, { useState } from 'react';
import jsPDF from 'jspdf';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Download, FileText, Search } from 'lucide-react';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

interface AnalysisRecord {
  id: number;
  fileName: string;
  type: 'Image' | 'Video' | 'AI-Check';
  result: 'Real' | 'Fake' | 'AI-Generated' | 'Natural';
  confidence: number;
  date: string;
  time: string;
}

interface ResultsPageProps {
  analyses: AnalysisRecord[];
}

export function ResultsPage({ analyses }: ResultsPageProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [filterResult, setFilterResult] = useState('all');
  const analysisRecords: AnalysisRecord[] = analyses;

  const filteredRecords = analysisRecords.filter((record) => {
    const matchesSearch = record.fileName.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType = filterType === 'all' || record.type.toLowerCase() === filterType;
    // Handle result filtering - group Real/Natural and Fake/AI-Generated
    let matchesResult = filterResult === 'all';
    if (filterResult === 'real') {
      matchesResult = record.result === 'Real' || record.result === 'Natural';
    } else if (filterResult === 'fake') {
      matchesResult = record.result === 'Fake' || record.result === 'AI-Generated';
    }
    return matchesSearch && matchesType && matchesResult;
  });

  const exportPDF = () => {
    const doc = new jsPDF({ unit: 'pt' });
    const marginLeft = 40;
    let y = 50;

    const addLine = (text: string, opts: { bold?: boolean; size?: number } = {}) => {
      if (opts.size) doc.setFontSize(opts.size);
      doc.setFont(undefined, opts.bold ? 'bold' : 'normal');
      doc.text(text, marginLeft, y);
      y += 16;
    };

    addLine('Analysis History', { bold: true, size: 16 });
    addLine(`Exported: ${new Date().toLocaleString()}`, { size: 11 });
    y += 6;

    if (filteredRecords.length === 0) {
      addLine('No records match the current filters.');
    } else {
      filteredRecords.forEach((record, idx) => {
        addLine(`${idx + 1}. ${record.fileName}`, { bold: true });
        addLine(`   Type: ${record.type}`);
        addLine(`   Result: ${record.result}`);
        addLine(`   Confidence: ${record.confidence}%`);
        addLine(`   Date: ${record.date} ${record.time}`);
        y += 6;
        if (y > 760) {
          doc.addPage();
          y = 40;
        }
      });
    }

    doc.save('analysis-history.pdf');
  };

  const exportCSV = () => {
    const rows = [
      ['File Name', 'Type', 'Result', 'Confidence (%)', 'Date', 'Time'],
      ...filteredRecords.map((r) => [
        r.fileName,
        r.type,
        r.result,
        r.confidence.toString(),
        r.date,
        r.time,
      ]),
    ];
    const csv = rows.map((r) => r.map((f) => `"${f.replace(/"/g, '""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'analysis-history.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="mb-6">
        <h1 className="text-foreground mb-2">Results & Reports</h1>
        <p className="text-muted-foreground">View and export previous analysis results</p>
      </div>

      {/* Filters and Search */}
      <Card className="shadow-md mb-6 bg-card border-border">
        <CardContent className="pt-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="md:col-span-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search by file name..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <Select value={filterType} onValueChange={setFilterType}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="image">Images</SelectItem>
                <SelectItem value="video">Videos</SelectItem>
                <SelectItem value="ai-check">AI-Check</SelectItem>
              </SelectContent>
            </Select>
            <Select value={filterResult} onValueChange={setFilterResult}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by result" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Results</SelectItem>
                <SelectItem value="real">Real/Natural</SelectItem>
                <SelectItem value="fake">Fake/AI-Generated</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Results Table */}
      <Card className="shadow-md mb-6 bg-card border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-foreground">Analysis History</CardTitle>
              <CardDescription>
                {filteredRecords.length} results found
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={exportCSV}>
                <FileText className="w-4 h-4 mr-2" />
                Export CSV
              </Button>
              <Button className="bg-primary hover:bg-primary/90" onClick={exportPDF}>
                <Download className="w-4 h-4 mr-2" />
                Export PDF
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="border border-border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="bg-secondary/50">
                  <TableHead className="text-foreground">File Name</TableHead>
                  <TableHead className="text-foreground">Type</TableHead>
                  <TableHead className="text-foreground">Result</TableHead>
                  <TableHead className="text-foreground">Confidence</TableHead>
                  <TableHead className="text-foreground">Date</TableHead>
                  <TableHead className="text-foreground">Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredRecords.map((record) => (
                  <TableRow key={record.id} className="hover:bg-secondary/50">
                    <TableCell className="text-foreground">{record.fileName}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{record.type}</Badge>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={record.result === 'Fake' || record.result === 'AI-Generated' ? 'destructive' : 'default'}
                        className={
                          record.result === 'Fake' 
                            ? '' 
                            : record.result === 'AI-Generated'
                              ? 'bg-purple-600 hover:bg-purple-700'
                              : 'bg-green-600 hover:bg-green-700'
                        }
                      >
                        {record.result}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-foreground">{record.confidence}%</TableCell>
                    <TableCell className="text-muted-foreground">{record.date}</TableCell>
                    <TableCell className="text-muted-foreground">{record.time}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="shadow-md bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-muted-foreground">Total Analyses</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-foreground">{analysisRecords.length}</p>
            <p className="text-xs text-muted-foreground mt-1">Last 7 days</p>
          </CardContent>
        </Card>
        <Card className="shadow-md bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-muted-foreground">Deepfakes Found</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-foreground">
              {analysisRecords.filter((r) => r.result === 'Fake').length}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {((analysisRecords.filter((r) => r.result === 'Fake').length / analysisRecords.length) * 100).toFixed(1)}% of total
            </p>
          </CardContent>
        </Card>
        <Card className="shadow-md bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm text-muted-foreground">Average Confidence</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-foreground">
              {(analysisRecords.reduce((acc, r) => acc + r.confidence, 0) / analysisRecords.length).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground mt-1">Across all analyses</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
